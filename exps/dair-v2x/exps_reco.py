# Copyright (c) Megvii Inc. All rights reserved.
from argparse import ArgumentParser, Namespace

import os
import mmcv
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import MultiStepLR

from dataset.nusc_mv_det_dataset import NuscMVDetDataset, collate_fn
from evaluators.det_evaluators import RoadSideEvaluator
from models.reco import RECO
from utils.torch_dist import all_gather_object, get_rank, synchronize
from utils.backup_files import backup_codebase
from utils.yaw_loss import *

H = 1080
W = 1920
final_dim = (864, 1536)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

data_root = "data/dair-v2x-i/"
gt_label_path = "data/dair-v2x-i-kitti/training/label_2"

backbone_conf = {
    'x_bound': [0, 140.8, 0.8],
    'y_bound': [-70.4, 70.4, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [-2.0, 0.0, 90],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'height_net_conf':
    dict(in_channels=512, mid_channels=512,
        # y_compensator=dict(
        #         enable=True,
        #         num_strips=6,
        #         out_dim=2,      # dx, dy
        #         scale=1.0,
        #     )
    )
}
ida_aug_conf = {
    'final_dim':
    final_dim,
    'H':
    H,
    'W':
    W,
    'bot_pct_lim': (0.0, 0.0),
    'cams': ['CAM_FRONT'],
    'Ncams': 1,
}

bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[0.0, -70.4, -10.0, 140.8, 70.4, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[0, -70.4, -5, 140.8, 70.4, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[0, -70.4, -5, 140.8, 70.4, 3],
    grid_size=[704, 704, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[0.0, -70.4, -10.0, 140.8, 70.4, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}


class RECOLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                 gpus: int = 1,
                 data_root=data_root,
                 eval_interval=1,
                 batch_size_per_device=8,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 default_root_dir='outputs/',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.basic_lr_per_img = 2e-4 / 64
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.ida_aug_conf = ida_aug_conf
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = RoadSideEvaluator(class_names=self.class_names,
                                           current_classes=["Car", "Pedestrian", "Cyclist"],
                                           data_root=data_root,
                                           gt_label_path=gt_label_path,
                                           output_dir=self.default_root_dir)
        self.model = RECO(self.backbone_conf, self.head_conf)
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.up_stride = 8
        self.downsample_factor = self.backbone_conf['downsample_factor'] // self.up_stride
        self.dbound = self.backbone_conf['d_bound']
        self.height_channels = int(self.dbound[2])

        self.automatic_optimization = False

    def forward(self, sweep_imgs, mats):
        return self.model(sweep_imgs, mats)

    def pad_boxes7(gt_boxes_list):
        # gt_boxes_list: list length B, each is (Mi,7)
        B = len(gt_boxes_list)
        Mmax = max(b.shape[0] for b in gt_boxes_list)
        device = gt_boxes_list[0].device
        dtype = gt_boxes_list[0].dtype
        boxes = torch.zeros((B, Mmax, 7), device=device, dtype=dtype)
        mask  = torch.zeros((B, Mmax), device=device, dtype=torch.bool)
        for i,b in enumerate(gt_boxes_list):
            m = b.shape[0]
            boxes[i, :m] = b
            mask[i, :m] = True
        return boxes, mask

    def pad_teacher_boxes(teacher_boxes_list, K=50, device=None, dtype=torch.float32):
        # teacher_boxes_list: length B*N, each is (Mi,4) xyxy
        BN = len(teacher_boxes_list)
        out = torch.zeros((BN, K, 4), device=device, dtype=dtype)
        valid = torch.zeros((BN, K), device=device, dtype=torch.bool)
        for i, b in enumerate(teacher_boxes_list):
            m = min(b.shape[0], K)
            if m > 0:
                out[i, :m] = b[:m]
                valid[i, :m] = True
        return out, valid


    # training with update yaw,pitch,roll,x,y,z
    def training_step(self, batch):
        opt_main, opt_yaw = self.optimizers()

        (sweep_imgs, mats, _, metas, gt_boxes, gt_labels) = batch

        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda(non_blocking=True)
            sweep_imgs = sweep_imgs.cuda(non_blocking=True)
            gt_boxes = [gt_box.cuda(non_blocking=True) for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda(non_blocking=True) for gt_label in gt_labels]

        preds = self(sweep_imgs, mats)

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        backbone = model.backbone

        # ---------------------- (A) Detection loss ----------------------
        gt_boxes_3d = [b[..., :9] for b in gt_boxes]  # list[(Mi,9)]
        targets = model.get_targets(gt_boxes_3d, gt_labels)
        detection_loss = model.loss(targets, preds)

        # ---------------------- (B) Pose reprojection loss ----------------------
        loss_pose_reproj = None

        if hasattr(backbone, "_cache_dpose2") and hasattr(backbone, "_cache_boundaries"):
            pose2 = backbone._cache_dpose2               # preferred (B,N,2,6)
            boundaries = backbone._cache_boundaries     # single boundary b1

            sensor2ego_base = backbone._cache_sensor2ego   # (B,N,4,4)
            K = backbone._cache_intrin                     # (B,N,3,3)
            ida = backbone._cache_ida                      # (B,N,4,4) or None

            # ---- pack GT ----
            gt3d_list, gt2d_list = [], []
            for b in gt_boxes:
                gt3d_list.append(b[..., :7].to(pose2.dtype))     # (x,y,z,dx,dy,dz,yaw)
                gt2d_list.append(b[..., 9:13].to(pose2.dtype))   # (xmin,ymin,xmax,ymax)

            gt3d_ego, gt_mask = self.pad_tensor_list(
                gt3d_list, pad_value=0.0, device=pose2.device, dtype=pose2.dtype
            )  # (B,M,7)
            gt2d_obs, _ = self.pad_tensor_list(
                gt2d_list, pad_value=-1.0, device=pose2.device, dtype=pose2.dtype
            )  # (B,M,4)

            # ---- unify pose2 -> (B,N,2,6) ----
            # 支持：(B,N,2,6) / (B,2,6) / (B,N,6) / (B,6)
            N = sensor2ego_base.shape[1]
            if pose2.dim() == 4 and pose2.size(-1) == 6:                  # (B,N,2,6)
                B = pose2.size(0)
                if pose2.size(1) != N:
                    if pose2.size(1) == 1 and N > 1:
                        pose2 = pose2.expand(B, N, pose2.size(2), 6).contiguous()
                    else:
                        raise RuntimeError(f"pose2 shape {pose2.shape} mismatch N={N}")
                if pose2.size(2) != 2:
                    raise RuntimeError(f"pose2 expected 2 segments, got {pose2.size(2)}")
            elif pose2.dim() == 3 and pose2.size(-1) == 6:                # (B,?,6)
                B = pose2.size(0)
                if pose2.size(1) == 2:  # (B,2,6) -> (B,N,2,6)
                    pose2 = pose2[:, None, :, :].expand(B, N, 2, 6).contiguous()
                elif pose2.size(1) == N:  # (B,N,6) -> treat as single seg, duplicate to 2 seg
                    pose2 = pose2[:, :, None, :].expand(B, N, 2, 6).contiguous()
                else:
                    raise RuntimeError(f"pose2 shape {pose2.shape} not recognized for N={N}")
            elif pose2.dim() == 2 and pose2.size(-1) == 6:                # (B,6)
                B = pose2.size(0)
                pose2 = pose2[:, None, None, :].expand(B, N, 2, 6).contiguous()
            else:
                raise RuntimeError(f"bad pose2 shape: {pose2.shape}")

            B = pose2.size(0)
            M = gt3d_ego.shape[1]

            # ---- unify boundary b1 -> (B,N) ----
            if boundaries.dim() == 1:           # (B,)
                b1_bn = boundaries[:, None]     # (B,1)
            elif boundaries.dim() == 2:         # (B,N) or (B,1)
                b1_bn = boundaries
            elif boundaries.dim() == 3:         # (B,N,1)
                b1_bn = boundaries[..., 0]
            else:
                raise RuntimeError(f"bad boundaries shape: {boundaries.shape}")

            if b1_bn.size(1) == 1 and N > 1:
                b1_bn = b1_bn.expand(B, N).contiguous()
            elif b1_bn.size(1) != N:
                raise RuntimeError(f"b1_bn shape {b1_bn.shape} mismatch N={N}")

            # ---- valid 2D ----
            valid2d = (gt2d_obs[..., 0] >= 0) & (gt2d_obs[..., 2] > gt2d_obs[..., 0]) & (gt2d_obs[..., 3] > gt2d_obs[..., 1])
            valid_obj = gt_mask & valid2d  # (B,M)

            if M > 0 and valid_obj.any():
                # ---- range gate near/far ----
                r_obj = torch.sqrt(gt3d_ego[..., 0] ** 2 + gt3d_ego[..., 1] ** 2)  # (B,M)
                r_obj = r_obj[:, None, :].expand(B, N, M)                          # (B,N,M)

                b1 = b1_bn.unsqueeze(-1)                                           # (B,N,1)
                tau = float(getattr(backbone, "range_yaw_tau", 2.0))

                g = torch.sigmoid((r_obj - b1) / tau)                              # (B,N,M)
                w_near = 1.0 - g
                w_far  = g
                wsum = w_near + w_far
                w_near = w_near / (wsum + 1e-6)
                w_far  = w_far  / (wsum + 1e-6)

                # ---- per-object pose (B,N,M,6): near/far blend ----
                pose_obj = (
                    w_near[..., None] * pose2[:, :, None, 0, :] +
                    w_far[..., None]  * pose2[:, :, None, 1, :]
                )  # (B,N,M,6)

                # ---- average over valid objects -> pose_bn (B,N,6) ----
                obj_w = valid_obj[:, None, :].float()  # (B,1,M)
                obj_w = obj_w.expand(B, N, M)          # (B,N,M)

                pose_bn = (pose_obj * obj_w[..., None]).sum(dim=2) / (obj_w.sum(dim=2)[..., None] + 1e-6)  # (B,N,6)

                # ---- build rotation from yaw/pitch/roll (Z-Y-X) ----
                yaw   = pose_bn[..., 0]
                pitch = pose_bn[..., 1]
                roll  = pose_bn[..., 2]
                t_delta = pose_bn[..., 3:6]  # (B,N,3)

                def rotx(a):
                    c = torch.cos(a); s = torch.sin(a)
                    R = torch.zeros((*a.shape, 3, 3), device=a.device, dtype=a.dtype)
                    R[..., 0, 0] = 1.0
                    R[..., 1, 1] = c;   R[..., 1, 2] = -s
                    R[..., 2, 1] = s;   R[..., 2, 2] = c
                    return R

                def roty(a):
                    c = torch.cos(a); s = torch.sin(a)
                    R = torch.zeros((*a.shape, 3, 3), device=a.device, dtype=a.dtype)
                    R[..., 1, 1] = 1.0
                    R[..., 0, 0] = c;   R[..., 0, 2] = s
                    R[..., 2, 0] = -s;  R[..., 2, 2] = c
                    return R

                def rotz(a):
                    c = torch.cos(a); s = torch.sin(a)
                    R = torch.zeros((*a.shape, 3, 3), device=a.device, dtype=a.dtype)
                    R[..., 2, 2] = 1.0
                    R[..., 0, 0] = c;   R[..., 0, 1] = -s
                    R[..., 1, 0] = s;   R[..., 1, 1] = c
                    return R

                R_delta = rotz(yaw) @ roty(pitch) @ rotx(roll)  # (B,N,3,3)

                def add_pose_on_sensor2ego(sensor2ego, R_delta, t_delta):
                    """
                    Left-multiply correction:
                        T' = [RΔ tΔ;0 1] @ T
                    sensor2ego: (B,N,4,4)
                    R_delta:   (B,N,3,3)
                    t_delta:   (B,N,3)  (in ego frame!)
                    """
                    T = sensor2ego.clone()
                    R0 = sensor2ego[..., :3, :3]
                    t0 = sensor2ego[..., :3, 3]
                    T[..., :3, :3] = R_delta @ R0
                    T[..., :3, 3]  = (R_delta @ t0.unsqueeze(-1)).squeeze(-1) + t_delta
                    return T

                sensor2ego_pose = add_pose_on_sensor2ego(sensor2ego_base, R_delta, t_delta)  # (B,N,4,4)

                # ---- project 3D corners -> 2D bbox ----
                corners = boxes3d_to_corners_ego(gt3d_ego)  # (B,M,8,3)
                pred2d, valid3d = project_corners_to_bbox2d(corners, sensor2ego_pose, K, ida=ida)

                gt2d_obs_bn = gt2d_obs[:, None, :, :].expand(B, N, M, 4).contiguous()
                valid_bn = valid_obj[:, None, :].expand(B, N, M).contiguous()
                mask = (valid_bn & valid3d).float()

                tokens = [meta['token'] for meta in metas]
                
                if 'image/008819.jpg' in tokens:
                    with torch.no_grad():
                        pred2d0, v0 = project_corners_to_bbox2d(corners, sensor2ego_base, K, ida=ida)
                        pred2d1, v1 = project_corners_to_bbox2d(corners, sensor2ego_pose, K, ida=ida)

                        m0 = (valid_obj[:,None,:].expand_as(v0) & v0).float().unsqueeze(-1)
                        m1 = (valid_obj[:,None,:].expand_as(v1) & v1).float().unsqueeze(-1)

                        base_err = ((pred2d0 - gt2d_obs_bn).abs() * m0).sum() / (m0.sum()*4 + 1e-6)
                        pose_err = ((pred2d1 - gt2d_obs_bn).abs() * m1).sum() / (m1.sum()*4 + 1e-6)

                    improve = base_err - pose_err
                    with open("hardcom150.log", 'a')as f:
                        f.write(f"improve_px_mean : {improve}\n")

                loss_per = F.smooth_l1_loss(pred2d, gt2d_obs_bn, reduction="none").sum(dim=-1)  # (B,N,M)
                loss_pose_reproj = (loss_per * mask).sum() / (mask.sum() + 1e-6)

                self.log("loss_pose_reproj", loss_pose_reproj, prog_bar=True)

        # ---------------------- (C) Optimizer steps ----------------------
        def set_requires_grad(module, flag: bool):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad_(flag)

        # (1) main update: freeze pose/z nets backbone['range_pose_net']
        if loss_pose_reproj is not None:
            set_requires_grad(getattr(backbone, "range_pose_net", None), False)
            set_requires_grad(getattr(backbone, "range_yaw_net",  None), False)  
            set_requires_grad(getattr(backbone, "range_z_net",    None), False)

        opt_main.zero_grad(set_to_none=True)
        self.manual_backward(detection_loss)
        opt_main.step()

        # (2) pose update: only pose reproj loss
        if loss_pose_reproj is not None:
            set_requires_grad(getattr(backbone, "range_pose_net", None), True)

            opt_yaw.zero_grad(set_to_none=True)
            self.manual_backward(loss_pose_reproj)

            # clip
            params = []
            for pg in opt_yaw.param_groups:
                params += [p for p in pg["params"] if p.grad is not None]
            if len(params) > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            pose_head = model.backbone.range_pose_net.pose_head  
            w0 = pose_head.weight.detach().clone()
            opt_yaw.step()

        self.log("detection_loss", detection_loss, prog_bar=True)
        return detection_loss


    
    def pad_tensor_list(self, tensor_list, pad_value=0.0, dim_last=None, device=None, dtype=None):
        """
        tensor_list: list of (Mi, D)
        return:
            out: (B, Mmax, D)
            mask: (B, Mmax) bool, True for valid
        """
        B = len(tensor_list)
        Mmax = max([t.shape[0] for t in tensor_list]) if B > 0 else 0
        if dim_last is None:
            dim_last = tensor_list[0].shape[-1] if Mmax > 0 else 0

        out = torch.full((B, Mmax, dim_last), pad_value, device=device, dtype=dtype)
        mask = torch.zeros((B, Mmax), device=device, dtype=torch.bool)

        for i, t in enumerate(tensor_list):
            if t.numel() == 0:
                continue
            m = t.shape[0]
            out[i, :m] = t
            mask[i, :m] = True
        return out, mask

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def test_epoch_end(self, test_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def configure_optimizers(self):
        
        lr_main = self.basic_lr_per_img * self.batch_size_per_device * self.gpus

        main_params = []
        import itertools
        yaw_params = set(map(
            id,
            itertools.chain(
                self.model.backbone.range_pose_net.parameters(),
                # self.model.backbone.range_yaw_net.parameters(),
                # self.model.backbone.range_z_net.parameters(),
            )
        ))

        for p in self.model.parameters():
            if id(p) not in yaw_params:
                main_params.append(p)

        opt_main = torch.optim.AdamW(
            main_params,
            lr=lr_main,
            weight_decay=1e-7,
        )

        opt_yaw = torch.optim.AdamW(
            self.model.backbone.range_pose_net.parameters(),
            lr=1e-6,
            weight_decay=1e-3,
        )

        sch_main = MultiStepLR(opt_main, [19, 23])
        sch_yaw  = MultiStepLR(opt_yaw,  [19, 23])

        return (
            [opt_main, opt_yaw],
            [sch_main, sch_yaw],
        )


    def train_dataloader(self):
        train_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_path=os.path.join(data_root, 'dair_12hz_infos_train.pkl'),
            is_train=True,
            use_cbgs=self.data_use_cbgs,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=False,
        )
        from functools import partial

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=False),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_path=os.path.join(data_root, 'dair_12hz_infos_val.pkl'),
            is_train=False,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser

def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    print(args)
    
    model = RECOLightningModel(**vars(args))
    checkpoint_callback = ModelCheckpoint(dirpath='./outputs/reco/checkpoints', filename='{epoch}', every_n_epochs=5, save_last=True, save_top_k=-1)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    if args.evaluate:
        # for ckpt_name in os.listdir(args.ckpt_path):
        # model_pth = os.path.join(args.ckpt_path, 'z_yaw_pitch_roll.ckpt')
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        backup_codebase(os.path.join('./outputs/reco', 'backup'))
        trainer.fit(model)
        
def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = RECOLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=100,
        accelerator='ddp',
        num_sanity_val_steps=0,
        # gradient_clip_val=5,
        limit_val_batches=0,
        enable_checkpointing=True,
        precision=32,
        default_root_dir='./outputs/reco')
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    run_cli()
