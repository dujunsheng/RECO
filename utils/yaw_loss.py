import torch
import torch.nn.functional as F

def boxes3d_to_corners_ego(boxes: torch.Tensor) -> torch.Tensor:

    # boxes: (B,M,7) [x,y,z,dx,dy,dz,yaw]
    x, y, z, dx, dy, dz, yaw = boxes.unbind(-1)
    B, M = x.shape
    device, dtype = boxes.device, boxes.dtype

    base = torch.tensor([
        [ 0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], device=device, dtype=dtype)  # (8,3)

    corners = base.view(1,1,8,3).repeat(B,M,1,1)
    corners[..., 0] *= dx.unsqueeze(-1)
    corners[..., 1] *= dy.unsqueeze(-1)
    corners[..., 2] *= dz.unsqueeze(-1)

    c = torch.cos(yaw); s = torch.sin(yaw)
    R = torch.zeros((B,M,3,3), device=device, dtype=dtype)
    R[...,0,0]=c; R[...,0,1]=-s
    R[...,1,0]=s; R[...,1,1]=c
    R[...,2,2]=1.0

    corners = corners @ R.transpose(-1,-2)
    corners[..., 0] += x.unsqueeze(-1)
    corners[..., 1] += y.unsqueeze(-1)
    corners[..., 2] += z.unsqueeze(-1)

    return corners  # (B,M,8,3)

def rotz(dyaw):
    c = torch.cos(dyaw); s = torch.sin(dyaw)
    R = torch.zeros((*dyaw.shape,3,3), device=dyaw.device, dtype=dyaw.dtype)
    R[...,0,0]=c; R[...,0,1]=-s
    R[...,1,0]=s; R[...,1,1]=c
    R[...,2,2]=1.0
    return R

def add_yaw_on_sensor2ego(sensor2ego, dyaw_bn):
    # sensor2ego: (B,N,4,4) sensor->ego
    T = sensor2ego.clone()
    Rz = rotz(dyaw_bn)                  # (B,N,3,3)
    R  = T[..., :3, :3]
    t  = T[..., :3, 3]
    T[..., :3, :3] = Rz @ R
    T[..., :3, 3]  = (Rz @ t.unsqueeze(-1)).squeeze(-1)
    return T

def invert_se3_rt(sensor2ego):
    R = sensor2ego[..., :3, :3]
    t = sensor2ego[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    return R_inv, t_inv   # ego->sensor: x_cam = R_inv x_ego + t_inv

def project_corners_to_bbox2d(corners_ego, sensor2ego_yaw, K, ida=None, eps=1e-4):
    """
    corners_ego: (B,M,8,3)
    sensor2ego_yaw: (B,N,4,4) sensor->ego (already yaw compensated)
    K: (B,N,3,3)
    ida: (B,N,4,4) optional (post augmentation)
    return pred_bbox2d: (B,N,M,4), valid_mask: (B,N,M)
    """
    B, M = corners_ego.shape[:2]
    N = sensor2ego_yaw.shape[1]
    device, dtype = corners_ego.device, corners_ego.dtype

    R_inv, t_inv = invert_se3_rt(sensor2ego_yaw)  # (B,N,3,3), (B,N,3)

    p = corners_ego[:, None, :, :, :].repeat(1, N, 1, 1, 1)  # (B,N,M,8,3)
    # p_cam = (R_inv.unsqueeze(-3) @ p.unsqueeze(-1)).squeeze(-1) + t_inv.unsqueeze(-2)  # (B,N,M,8,3)
    p_cam = (R_inv[:, :, None, None, :, :] @ p[..., None]).squeeze(-1) \
        + t_inv[:, :, None, None, :]


    X, Y, Z = p_cam[...,0], p_cam[...,1], p_cam[...,2]
    valid = Z > eps
    box_valid = (valid.sum(dim=-1) >= 4)  # (B,N,M)

    x = X / (Z + eps)
    y = Y / (Z + eps)

    fx = K[...,0,0].unsqueeze(-1).unsqueeze(-1)
    fy = K[...,1,1].unsqueeze(-1).unsqueeze(-1)
    cx = K[...,0,2].unsqueeze(-1).unsqueeze(-1)
    cy = K[...,1,2].unsqueeze(-1).unsqueeze(-1)

    u = fx * x + cx
    v = fy * y + cy

    if ida is not None:
        a00 = ida[...,0,0].unsqueeze(-1).unsqueeze(-1)
        a01 = ida[...,0,1].unsqueeze(-1).unsqueeze(-1)
        a03 = ida[...,0,3].unsqueeze(-1).unsqueeze(-1)
        a10 = ida[...,1,0].unsqueeze(-1).unsqueeze(-1)
        a11 = ida[...,1,1].unsqueeze(-1).unsqueeze(-1)
        a13 = ida[...,1,3].unsqueeze(-1).unsqueeze(-1)
        u2 = a00*u + a01*v + a03
        v2 = a10*u + a11*v + a13
        u, v = u2, v2

    BIG = torch.tensor(1e9, device=device, dtype=dtype)
    NEG = torch.tensor(-1e9, device=device, dtype=dtype)

    u_min = torch.where(valid, u, BIG).min(dim=-1).values
    v_min = torch.where(valid, v, BIG).min(dim=-1).values
    u_max = torch.where(valid, u, NEG).max(dim=-1).values
    v_max = torch.where(valid, v, NEG).max(dim=-1).values

    pred = torch.stack([u_min, v_min, u_max, v_max], dim=-1)  # (B,N,M,4)

    return pred, box_valid

def soft_range_weights(r, b1, b2, b3, tau=2.0):
    g1 = torch.sigmoid((r - b1) / tau)
    g2 = torch.sigmoid((r - b2) / tau)
    g3 = torch.sigmoid((r - b3) / tau)
    w0 = 1 - g1
    w1 = g1 - g2
    w2 = g2 - g3
    w3 = g3
    w = torch.stack([w0,w1,w2,w3], dim=-1).clamp(min=0.0)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
    return w  # (...,4)
