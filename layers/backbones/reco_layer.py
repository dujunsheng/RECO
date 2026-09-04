# Copyright (c) Megvii Inc. All rights reserved.
import numpy as np

import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from ops.voxel_pooling import voxel_pooling
import math

__all__ = ['RECOLSSFPN']


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class HeightNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 height_channels):
        super(HeightNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.height_mlp = Mlp(27, mid_channels, mid_channels)
        self.height_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.height_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            
        )
        self.height_layer = nn.Conv2d(mid_channels,
                      height_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)

    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        height_se = self.height_mlp(mlp_input)[..., None, None]
        height = self.height_se(x, height_se)
        height = self.height_conv(height)
        height = self.height_layer(height)
        return torch.cat([height, context], dim=1)

class RangeZOffsetNet(nn.Module):
    """
    Predict 3 boundaries (=>4 segments) and per-segment dz offsets.
    Output is per-camera (B, N, ...).
    """
    def __init__(self, feat_dim, cam_param_dim=27, hidden=128,
                 base_boundaries=(20.0, 40.0, 80.0),
                 max_delta=10.0, dz_max=0.5):
        super().__init__()
        self.register_buffer("base_b", torch.tensor(base_boundaries, dtype=torch.float32))
        self.max_delta = float(max_delta)
        self.dz_max = float(dz_max)
        self.bn = nn.BatchNorm1d(feat_dim + cam_param_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + cam_param_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.delta_b = nn.Linear(hidden, 3)  # residual for boundaries
        self.dz_head  = nn.Linear(hidden, 1)

        nn.init.zeros_(self.delta_b.weight); nn.init.zeros_(self.delta_b.bias)
        nn.init.zeros_(self.dz_head.weight); nn.init.zeros_(self.dz_head.bias)

    def forward(self, feat_vec, cam_param):
        x = torch.cat([feat_vec, cam_param], dim=-1)
        x = self.bn(x)
        x = self.mlp(x)

        # dz4 = torch.tanh(self.dz_head(x)) * self.dz_max  # (B*N,4)
        dz4 = self.dz_head(x)

        # residual boundary: [-max_delta, +max_delta]
        delta = torch.tanh(self.delta_b(x)) * self.max_delta  # (B*N,3)
        b = self.base_b[None, :].to(delta.device) + delta     # (B*N,3)

        # enforce monotonic increasing with sorting (simple and effective)
        b_sorted, _ = torch.sort(b, dim=-1)
        return b_sorted, dz4

class RangeYawOffsetNet(nn.Module):
    """
    Predict 1 boundary (=>2 segments) and per-segment yaw offsets (dyaw2).
    Input expected as (B*N, feat_dim) and (B*N, cam_param_dim).
    Output: boundary (B*N,1), dyaw2 (B*N,2)
    """
    def __init__(self, feat_dim, cam_param_dim=27, hidden=128,
                 base_boundary=40.0,
                 max_delta=10.0, yaw_max_deg= 30,
                 b_min=1.0, b_max=200.0):
        super().__init__()
        self.register_buffer("base_b", torch.tensor([base_boundary], dtype=torch.float32))  # (1,)
        self.max_delta = float(max_delta)

        self.yaw_max = float(yaw_max_deg) * np.pi / 180.0

        # 如果你之前 BN 容易不稳，建议换 LayerNorm：nn.LayerNorm(feat_dim+cam_param_dim)
        self.bn = nn.BatchNorm1d(feat_dim + cam_param_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + cam_param_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        self.delta_b = nn.Linear(hidden, 1)  # residual for 1 boundary
        self.yaw_head = nn.Linear(hidden, 2) # 2 segments

        nn.init.zeros_(self.delta_b.weight); nn.init.zeros_(self.delta_b.bias)
        nn.init.zeros_(self.yaw_head.weight); nn.init.zeros_(self.yaw_head.bias)

        self.b_min = float(b_min)
        self.b_max = float(b_max)

    def forward(self, feat_vec, cam_param):
        # feat_vec: (B*N, feat_dim)  cam_param: (B*N, cam_param_dim)
        x = torch.cat([feat_vec, cam_param], dim=-1)  # (B*N, C)
        x = self.bn(x)
        x = self.mlp(x)

        # dyaw in [-yaw_max, +yaw_max], shape (B*N,2)
        dyaw2 = self.yaw_head(x)
        dyaw = dyaw2.clamp(-self.yaw_max, self.yaw_max)

        # boundary in meters: base + residual, shape (B*N,1)
        delta = torch.tanh(self.delta_b(x)) * self.max_delta  # (B*N,1)
        b = self.base_b[None, :].to(delta.device) + delta     # (B*N,1)

        # optional clamp to keep boundary in a sensible range
        b = b.clamp(self.b_min, self.b_max)

        with torch.no_grad():
            import math
            mean_rad = dyaw2.abs().mean().item()
            max_rad  = dyaw2.abs().max().item()
            rad2deg = 180.0 / math.pi

        return b, dyaw2

class RangePoseOffsetNet(nn.Module):
    """
    Predict 1 boundary (=> 2 segments) and per-segment 6DoF offsets:
      dpose2: (B*N, 2, 6) with order [yaw, pitch, roll, tx, ty, tz]
    Inputs:
      feat_vec: (B*N, feat_dim)
      cam_param: (B*N, cam_param_dim)
    Outputs:
      b: (B*N, 1) boundary (meters)
      dpose2: (B*N, 2, 6) offsets (rad, meters)
    """
    def __init__(self,
                 feat_dim,
                 cam_param_dim=27,
                 hidden=1024,
                 base_boundary=40.0,
                 max_delta_b=10.0,
                 # angle limits (deg)
                 yaw_max_deg=3.0,
                 pitch_max_deg=3.0,
                 roll_max_deg=3.0,
                 # translation limits (meters)
                 t_max_xyz=(0.2, 0.2, 0.1),
                 b_min=1.0,
                 b_max=200.0,
                 norm_type="bn"  # "bn" or "ln"
                 ):
        super().__init__()
        self.register_buffer("base_b", torch.tensor([base_boundary], dtype=torch.float32))
        self.max_delta_b = float(max_delta_b)

        # rad limits
        self.yaw_max   = float(yaw_max_deg)   * math.pi / 180.0
        self.pitch_max = float(pitch_max_deg) * math.pi / 180.0
        self.roll_max  = float(roll_max_deg)  * math.pi / 180.0

        tx, ty, tz = t_max_xyz
        self.tx_max = float(tx)
        self.ty_max = float(ty)
        self.tz_max = float(tz)

        in_dim = feat_dim + cam_param_dim  #  cam_param_dim
        if norm_type == "ln":
            self.norm = nn.LayerNorm(in_dim)
        else:
            self.norm = nn.BatchNorm1d(in_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # boundary residual
        self.delta_b = nn.Linear(hidden, 1)

        # 2 segments * 6DoF = 12 dims
        self.pose_head = nn.Linear(hidden, 12)

        # zero init => start from identity compensation
        nn.init.zeros_(self.delta_b.weight); nn.init.zeros_(self.delta_b.bias)
        nn.init.zeros_(self.pose_head.weight); nn.init.zeros_(self.pose_head.bias)

        self.b_min = float(b_min)
        self.b_max = float(b_max)

    def forward(self, feat_vec, cam_param, debug=True):
        x = torch.cat([cam_param, feat_vec], dim=-1)  #  [feat_vec, cam_param](B*N, C) [cam_param]
        x = self.norm(x)
        x = self.mlp(x)

        # ---- boundary ----
        delta = torch.tanh(self.delta_b(x)) * self.max_delta_b           # (B*N,1)
        b = self.base_b[None, :].to(delta.device) + delta                # (B*N,1)
        b = b.clamp(self.b_min, self.b_max)

        # ---- 6DoF for 2 segments ----
        raw = self.pose_head(x).view(-1, 2, 6)                           # (B*N,2,6)

        # Use tanh scaling (smooth) instead of clamp (hard)
        dyaw   = torch.tanh(raw[..., 0]) * self.yaw_max
        dpitch = torch.tanh(raw[..., 1]) * self.pitch_max
        droll  = torch.tanh(raw[..., 2]) * self.roll_max

        dtx = torch.tanh(raw[..., 3]) * self.tx_max
        dty = torch.tanh(raw[..., 4]) * self.ty_max
        dtz = torch.tanh(raw[..., 5]) * self.tz_max

        # dyaw   = raw[..., 0]
        # dpitch = raw[..., 1]
        # droll  = raw[..., 2]

        # dtx = raw[..., 3]
        # dty = raw[..., 4]
        # dtz = raw[..., 5]

        dpose2 = torch.stack([dyaw, dpitch, droll, dtx, dty, dtz], dim=-1)  # (B*N,2,6)

        return b, dpose2

class RECOLSSFPN(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, height_net_conf,
                 range_z_extrinsics=None,   
                 ):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            height_net_conf (dict): Config for height net.
        """

        super(RECOLSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.height_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.height_net = self._configure_height_net(height_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

        self.range_z_extrinsics = range_z_extrinsics or []

        self.learn_range_z = False  # 
        out_ch = img_neck_conf.get("out_channels", 256)
        if isinstance(out_ch, (list, tuple)):
            self._cam_feat_dim = int(sum(out_ch))   # [128,128,128,128] -> 512
        else:
            self._cam_feat_dim = int(out_ch)

        self.range_z_net = RangeZOffsetNet(
            feat_dim=self.height_channels,  
            cam_param_dim=27,
            hidden=128,
            # max_range=120.0,   
            dz_max=0.5,       
        )
        self.learn_range_yaw = True
        self.range_yaw_net = RangeYawOffsetNet(
            feat_dim=self.height_channels,
            cam_param_dim=27,
            hidden=128,
            yaw_max_deg=5.0,   
        )

        # self.learn_range_pose = True
        self.range_pose_net = RangePoseOffsetNet(
            feat_dim=self.height_channels,
            cam_param_dim=27,
            hidden=128,
            yaw_max_deg=5.0,   
        )

    def _configure_height_net(self, height_net_conf):
        return HeightNet(
            height_net_conf['in_channels'],
            height_net_conf['mid_channels'],
            self.output_channels,
            self.height_channels,
        )

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        
        # DID
        alpha = 1.5
        d_coords = np.arange(self.d_bound[2]) / self.d_bound[2]
        d_coords = np.power(d_coords, alpha)
        d_coords = self.d_bound[0] + d_coords * (self.d_bound[1] - self.d_bound[0])
        d_coords = torch.tensor(d_coords, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum
    
    def height2localtion(self, points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        reference_heights = reference_heights.view(batch_size, num_cams, 1, 1, 1, 1,
                                                   1).repeat(1, 1, points.shape[2], points.shape[3], points.shape[4], 1, 1)
        height = -1 * points[:, :, :, :, :, 2, :] + reference_heights[:, :, :, :, :, 0, :]
        
        points_const = points.clone()
        points_const[:, :, :, :, :, 2, :] = 10
        points_const = torch.cat(
            (points_const[:, :, :, :, :, :2] * points_const[:, :, :, :, :, 2:3],
             points_const[:, :, :, :, :, 2:]), 5)
        combine_virtual = sensor2virtual_mat.matmul(torch.inverse(intrin_mat))
        points_virtual = combine_virtual.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points_const)
        ratio = height[:, :, :, :, :, 0] / points_virtual[:, :, :, :, :, 1, 0]
        ratio = ratio.view(batch_size, num_cams, ratio.shape[2], ratio.shape[3], ratio.shape[4], 1, 1).repeat(1, 1, 1, 1, 1, 4, 1)
        points = points_virtual * ratio
        points[:, :, :, :, :, 3, :] = 1
        combine_ego = sensor2ego_mat.matmul(torch.inverse(sensor2virtual_mat))
        points = combine_ego.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        return points
    
    def get_geometry(self, sensor2ego_mat, sensor2virtual_mat, intrin_mat, ida_mat, reference_heights, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        points = self.height2localtion(points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights) 
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def _forward_height_net(self, feat, mats_dict):
        return self.height_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_height):
        return img_feat_with_height

    def _bev_warp_yaw(self, bev: torch.Tensor, dyaw_b: torch.Tensor):
        """
        bev: (B, C, H, W)
        dyaw_b: (B,) radians
        """
        B, C, H, W = bev.shape
        device, dtype = bev.device, bev.dtype

        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)

        # --- torch<1.10 compatibility: no indexing keyword ---
        # default is 'ij' for 2D in older torch, but we enforce shape by building and stacking explicitly
        grid_y, grid_x = torch.meshgrid(ys, xs)  # both (H, W)

        # grid_sample expects grid in (B, H, W, 2) with last dim = (x, y)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,H,W,2)

        c = torch.cos(dyaw_b).view(B, 1, 1)
        s = torch.sin(dyaw_b).view(B, 1, 1)

        x = grid[..., 0]
        y = grid[..., 1]
        x2 = c * x - s * y
        y2 = s * x + c * y
        grid2 = torch.stack([x2, y2], dim=-1)

        return F.grid_sample(
            bev, grid2,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              is_return_height=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_height (bool, optional): Whether to return height.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        height_feature = self._forward_height_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict,
        )
        height = height_feature[:, :self.height_channels].softmax(1)
        
        img_feat_with_height = height.unsqueeze(
            1) * height_feature[:, self.height_channels:(
                self.height_channels + self.output_channels)].unsqueeze(2)
        img_feat_with_height = self._forward_voxel_net(img_feat_with_height)

        img_feat_with_height = img_feat_with_height.reshape(
            batch_size,
            num_cams,
            img_feat_with_height.shape[1],
            img_feat_with_height.shape[2],
            img_feat_with_height.shape[3],
            img_feat_with_height.shape[4],
        )
        
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['sensor2virtual_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict['reference_heights'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )

        if getattr(self, "learn_range_z", False):

            # 1) per-cam global feature vector (from source_features)
            # source_features: (B, N, C, H, W)
            cam_feat = source_features  
            B, N, C, Hf, Wf = cam_feat.shape
            cam_feat_vec = F.adaptive_avg_pool2d(cam_feat.view(B*N, C, Hf, Wf), 1).view(B*N, C)

            # 2) camera param vector (B,N,27)
            cam_param = self._build_cam_param(mats_dict, sweep_index).reshape(B*N, -1)

            # 3) predict boundaries and dz
            _, dz4 = self.range_z_net(cam_feat_vec, cam_param)  # (B*N,3), (B*N,4)
            boundaries = boundaries.view(B, N, 3)  # meters
            dz4 = dz4.view(B, N, 4)               # meters

            # 4) compute ego planar range r0 from current geometry
            r0 = torch.sqrt(geom_xyz[..., 0] ** 2 + geom_xyz[..., 1] ** 2)  # (B,N,D,H,W)

            # 5) build 4 masks using predicted boundaries (per cam)
            boundaries = torch.sort(boundaries, dim=-1).values  # (B,N,3) 保证 b1<=b2<=b3
            b1 = boundaries[:, :, 0].view(B, N, 1, 1, 1)
            b2 = boundaries[:, :, 1].view(B, N, 1, 1, 1)
            b3 = boundaries[:, :, 2].view(B, N, 1, 1, 1)


            # print(f'boundaries: {boundaries}, yawl: {dyaw4}')

            # temperature: smaller => sharper (but too small may cause instability)
            tau = getattr(self, "range_z_tau", 5.0)  
            tau = float(tau)

            # g1,g2,g3 是“到达边界”的软指示：r0 越大越接近 1
            g1 = torch.sigmoid((r0 - b1) / tau)
            g2 = torch.sigmoid((r0 - b2) / tau)
            g3 = torch.sigmoid((r0 - b3) / tau)

            w0 = 1 - g1
            w1 = g1 - g2
            w2 = g2 - g3
            w3 = g3

            # 数值稳定：夹一下避免负数/和不为1（理论上单调时不会负）
            w = torch.stack([w0, w1, w2, w3], dim=-1).clamp(min=0.0)  # (B,N,D,H,W,4)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

            geom_mix = geom_xyz  # start from base geometry

            sensor2ego_base = mats_dict['sensor2ego_mats'][:, sweep_index, ...]      # (B,N,4,4)
            sensor2virtual  = mats_dict['sensor2virtual_mats'][:, sweep_index, ...] # (B,N,4,4)
            intrin          = mats_dict['intrin_mats'][:, sweep_index, ...]         # (B,N,4,4)
            ida             = mats_dict['ida_mats'][:, sweep_index, ...]            # (B,N,4,4)
            ref_h           = mats_dict['reference_heights'][:, sweep_index, ...]   # (B,N,?)
            bda             = mats_dict.get('bda_mat', None)

            # 只算一次
            inv_base = torch.linalg.inv(sensor2ego_base)  # (B,N,4,4)
            # helper to shift mats by dz (Tensor dz_bn: (B,N))
            def shift_from_inv_base(inv_base: torch.Tensor, dz_bn: torch.Tensor) -> torch.Tensor:
                """
                inv_base: inv(sensor2ego_base)  (B,N,4,4)
                dz_bn:    (B,N)
                return:   sensor2ego_shift (B,N,4,4)
                """
                T = inv_base.clone()
                T[..., 2, 3] = T[..., 2, 3] + dz_bn
                return torch.linalg.inv(T)

            # 4 geoms
            # dz4[:,:,k] : (B,N) 直接传
            sensor2ego_shift0 = shift_from_inv_base(inv_base, dz4[:, :, 0])
            geom0 = self.get_geometry(sensor2ego_shift0, sensor2virtual, intrin, ida, ref_h, bda)  # (B,N,D,H,W,3)

            sensor2ego_shift1 = shift_from_inv_base(inv_base, dz4[:, :, 1])
            geom1 = self.get_geometry(sensor2ego_shift1, sensor2virtual, intrin, ida, ref_h, bda)

            sensor2ego_shift2 = shift_from_inv_base(inv_base, dz4[:, :, 2])
            geom2 = self.get_geometry(sensor2ego_shift2, sensor2virtual, intrin, ida, ref_h, bda)

            sensor2ego_shift3 = shift_from_inv_base(inv_base, dz4[:, :, 3])
            geom3 = self.get_geometry(sensor2ego_shift3, sensor2virtual, intrin, ida, ref_h, bda)


            # -------- soft mixture (fully differentiable) --------
            w0 = w[..., 0].unsqueeze(-1)  # (B,N,D,H,W,1)
            w1 = w[..., 1].unsqueeze(-1)
            w2 = w[..., 2].unsqueeze(-1)
            w3 = w[..., 3].unsqueeze(-1)

            geom_xyz = w0 * geom0 + w1 * geom1 + w2 * geom2 + w3 * geom3

        if getattr(self, "learn_range_yaw", False):

            # cam_feat = source_features
            # B, N, C, Hf, Wf = cam_feat.shape
            # cam_feat_vec = F.adaptive_avg_pool2d(cam_feat.view(B*N, C, Hf, Wf), 1).view(B*N, C)

            # 使用高度特征
            # img_feat_with_height: (B,S,D,C,Hf,Wf)
            B, S, D, Ch, Hf, Wf = img_feat_with_height.shape

            # 聚合 depth 维 -> (B,S,Ch,Hf,Wf)
            hfeat = img_feat_with_height.mean(dim=2)

            # 如果你要按相机 N 来做（你这里 N=1），对齐成 (B,S,N,Ch,Hf,Wf)
            hfeat = hfeat.unsqueeze(2)  # (B,S,1,Ch,Hf,Wf)

            hfeat = hfeat[:, sweep_index]          # (B,1,Ch,Hf,Wf)
            B, N, Ch, Hf, Wf = hfeat.shape

            cam_feat_vec = F.adaptive_avg_pool2d(hfeat.view(B*N, Ch, Hf, Wf), 1).view(B*N, Ch)

            cam_param = self._build_cam_param(mats_dict, sweep_index).reshape(B*N, -1)

            boundaries, dpose2 = self.range_pose_net(cam_feat_vec.detach(), cam_param.detach())

            # 一个边界两个yaw
            boundaries = boundaries.view(B, N, 1)     # (B,N,1)
            # cache for later use (keep graph!)
            self._cache_dpose2 = dpose2.view(B, N, 2, 6)   # 方便你后续 loss/调试
            self._cache_boundaries = boundaries.view(B, N, 1)

            # cache current sweep mats needed for projection
            self._cache_sensor2ego = mats_dict['sensor2ego_mats'][:, sweep_index, ...]      # (B,N,4,4)
            self._cache_intrin = mats_dict['intrin_mats'][:, sweep_index, ..., :3, :3]      # (B,N,3,3)
            self._cache_ida = mats_dict['ida_mats'][:, sweep_index, ...]           

            # range r0
            r0 = torch.sqrt(geom_xyz[..., 0] ** 2 + geom_xyz[..., 1] ** 2)                  # (... same shape as geom_xyz[...,0])

            # NEW: single boundary
            b1 = boundaries[..., 0].view(B, N).contiguous()
            b1 = b1.view(B, N, 1, 1, 1)

            tau = float(getattr(self, "range_yaw_tau", 2))

            # NEW: 2-seg soft gating
            g = torch.sigmoid((r0 - b1) / tau)   # g≈0: near, g≈1: far
            w_near = (1.0 - g)
            w_far  = g

            # 可选：归一化（其实 w_near+w_far=1，不写也行）
            wsum = w_near + w_far
            w_near = w_near / (wsum + 1e-6)
            w_far  = w_far  / (wsum + 1e-6)

            sensor2ego_base = mats_dict['sensor2ego_mats'][:, sweep_index, ...]
            sensor2virtual  = mats_dict['sensor2virtual_mats'][:, sweep_index, ...]
            intrin          = mats_dict['intrin_mats'][:, sweep_index, ...]
            ida             = mats_dict['ida_mats'][:, sweep_index, ...]
            ref_h           = mats_dict['reference_heights'][:, sweep_index, ...]
            bda             = mats_dict.get('bda_mat', None)

            def rot_x(a):
                c, s = torch.cos(a), torch.sin(a)
                R = torch.zeros((*a.shape, 3, 3), device=a.device, dtype=a.dtype)
                R[..., 0, 0] = 1.0
                R[..., 1, 1] = c;   R[..., 1, 2] = -s
                R[..., 2, 1] = s;   R[..., 2, 2] = c
                return R

            def rot_y(a):
                c, s = torch.cos(a), torch.sin(a)
                R = torch.zeros((*a.shape, 3, 3), device=a.device, dtype=a.dtype)
                R[..., 1, 1] = 1.0
                R[..., 0, 0] = c;   R[..., 0, 2] = s
                R[..., 2, 0] = -s;  R[..., 2, 2] = c
                return R

            def rot_z(a):
                c, s = torch.cos(a), torch.sin(a)
                R = torch.zeros((*a.shape, 3, 3), device=a.device, dtype=a.dtype)
                R[..., 2, 2] = 1.0
                R[..., 0, 0] = c;   R[..., 0, 1] = -s
                R[..., 1, 0] = s;   R[..., 1, 1] = c
                return R

            def euler_zyx(yaw, pitch, roll):
                # 常用的 Z(Yaw) * Y(Pitch) * X(Roll)
                return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

            def apply_pose_on_sensor2ego(sensor2ego, dpose_bn):
                """
                sensor2ego: (B,N,4,4)  sensor -> ego
                dpose_bn:   (B,N,6)    [yaw,pitch,roll, tx,ty,tz] in (rad, m)
                return:     (B,N,4,4)  corrected sensor2ego
                说明：采用“左乘”补偿：T' = T_delta @ T0
                """
                T0 = sensor2ego
                T  = T0.clone()

                yaw   = dpose_bn[..., 0]
                pitch = dpose_bn[..., 1]
                roll  = dpose_bn[..., 2]
                dt    = dpose_bn[..., 3:6]                       # (B,N,3)

                R_delta = euler_zyx(yaw, pitch, roll)            # (B,N,3,3)
                R0 = T0[..., :3, :3]
                t0 = T0[..., :3, 3]

                # 左乘：R' = R_delta R0 ; t' = R_delta t0 + dt
                T[..., :3, :3] = R_delta @ R0
                T[..., :3, 3]  = (R_delta @ t0.unsqueeze(-1)).squeeze(-1) + dt
                return T

            # ---- unify dpose2 shape -> (B,N,2,6) ----
            dpose2 = dpose2.clone()
            if dpose2.dim() == 3 and dpose2.shape[-2:] == (2, 6):          # (B*N,2,6)
                dpose2 = dpose2.view(B, N, 2, 6)
            elif dpose2.dim() == 4 and dpose2.shape[-2:] == (2, 6):        # (B,N,2,6)
                if dpose2.size(1) == 1 and N > 1:
                    dpose2 = dpose2.expand(B, N, 2, 6).contiguous()
                elif dpose2.size(1) != N:
                    raise RuntimeError(f"dpose2 shape {dpose2.shape} mismatch N={N}")
            else:
                raise RuntimeError(f"bad dpose2 shape: {dpose2.shape}")

            # near / far pose: (B,N,6)
            dpose_near = dpose2[:, :, 0, :]   # [yaw,pitch,roll, tx,ty,tz]
            dpose_far  = dpose2[:, :, 1, :]

            # apply pose to sensor2ego
            sensor2ego_shift_near = apply_pose_on_sensor2ego(sensor2ego_base, dpose_near)  # (B,N,4,4)
            geom_near = self.get_geometry(sensor2ego_shift_near, sensor2virtual, intrin, ida, ref_h, bda)

            sensor2ego_shift_far  = apply_pose_on_sensor2ego(sensor2ego_base, dpose_far)   # (B,N,4,4)
            geom_far  = self.get_geometry(sensor2ego_shift_far, sensor2virtual, intrin, ida, ref_h, bda)

            # blend
            w_near_ = w_near.unsqueeze(-1)
            w_far_  = w_far.unsqueeze(-1)
            geom_xyz = w_near_ * geom_near + w_far_ * geom_far
            

        # ===================== end =====================

        img_feat_with_height = img_feat_with_height.permute(0, 1, 3, 4, 5, 2)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        
        feature_map = voxel_pooling(geom_xyz, img_feat_with_height.contiguous(),
                                   self.voxel_num.cuda())
        
        if is_return_height:
            return feature_map.contiguous(), height
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                is_return_height=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            is_return_height=is_return_height)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_height else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    is_return_height=False)
                ret_feature_list.append(feature_map)

        if is_return_height:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)

    def _build_cam_param(self, mats_dict, sweep_index):
        intrins = mats_dict['intrin_mats'][:, sweep_index:sweep_index+1, ..., :3, :3]
        ida = mats_dict['ida_mats'][:, sweep_index:sweep_index+1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, sweep_index:sweep_index+1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(intrins.shape[0], 1, 1, 4, 4).repeat(1, 1, intrins.shape[2], 1, 1)

        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(intrins.shape[0], 1, intrins.shape[2], -1),
            ],
            -1,
        )  # (B,1,N,27)

        mlp_input = mlp_input[:, 0]  # (B,N,27)
        return mlp_input