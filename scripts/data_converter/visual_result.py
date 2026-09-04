import os
import json
import numpy as np
import cv2
from visual_tools import project_to_image, draw_box_3d

# 复用你已有的：
# - project_to_image(pts_3d, P)
# - draw_box_3d(image, corners_2d, c)
# 这里假设你已经把它们定义好了

CLASSES = [
    'car','truck','construction_vehicle','bus','trailer','barrier',
    'motorcycle','bicycle','pedestrian','traffic_cone'
]
# 你可以按需改颜色
COLOR = {
    "car": (0,255,0),
    "truck": (0,255,255),
    "construction_vehicle": (255,0,255),
    "bus": (0,128,255),
    "trailer": (128,255,0),
    "barrier": (255,128,0),
    "motorcycle": (0,0,255),
    "bicycle": (255,0,0),
    "pedestrian": (255,255,0),
    "traffic_cone": (128,128,255),
}

def load_K_from_camera_intrinsic_json(path: str) -> np.ndarray:
    d = json.load(open(path, "r"))
    K = np.array(d["cam_K"], dtype=np.float32).reshape(3, 3)
    return K

def load_T_lidar_to_cam_from_json(path: str) -> np.ndarray:
    d = json.load(open(path, "r"))
    R = np.array(d["rotation"], dtype=np.float32).reshape(3, 3)
    t = np.array(d["translation"], dtype=np.float32).reshape(-1)
    # 兼容 [[x],[y],[z]]
    t = t[:3]
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def corners_3d_from_lidar_box(x, y, z, dx, dy, dz, yaw):
    """
    以 CenterPoint 常见定义：yaw 绕 Z 轴旋转（lidar/ego 坐标系）
    corners: (8,3) in lidar
    """
    # 8个角点相对中心
    # dx,dy,dz 是 box 尺寸（长宽高/或宽长高），不同实现可能 dx=长(x方向)、dy=宽(y方向)
    # 这里按常见：dx沿x，dy沿y，dz沿z
    hx, hy, hz = dx/2.0, dy/2.0, dz/2.0

    corners = np.array([
        [ hx,  hy,  hz],
        [ hx, -hy,  hz],
        [-hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy, -hz],
        [-hx, -hy, -hz],
        [-hx,  hy, -hz],
    ], dtype=np.float32)

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    corners = corners @ R.T
    corners += np.array([x, y, z], dtype=np.float32)
    return corners  # (8,3)

def transform_pts(pts_xyz: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    """pts_xyz: (N,3) -> (N,3)"""
    pts_h = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0],1), dtype=np.float32)], axis=1)
    out = (T_4x4 @ pts_h.T).T
    return out[:, :3]

def draw_centerpoint_preds_on_image(
    image_bgr: np.ndarray,
    bboxes_9: np.ndarray,   # (N,9): x,y,z,dx,dy,dz,yaw,vx,vy
    scores: np.ndarray,     # (N,)
    labels: np.ndarray,     # (N,) 0-based
    K: np.ndarray,          # (3,3)
    T_lidar_to_cam: np.ndarray,  # (4,4)
    score_thr: float = 0.3,
):
    """
    直接把预测 3D 框投影到图像并画线框
    """
    # P=[K|0]，因为 pts 已经在 camera 坐标系了
    P = np.zeros((3,4), dtype=np.float32)
    P[:3,:3] = K

    img = image_bgr.copy()

    for box, sc, lb in zip(bboxes_9, scores, labels):
        if float(sc) < score_thr:
            continue

        x, y, z, dx, dy, dz, yaw = map(float, box[:7])
        corners_lidar = corners_3d_from_lidar_box(x, y, z, dx, dy, dz, yaw)

        # lidar -> camera
        corners_cam = transform_pts(corners_lidar, T_lidar_to_cam)

        # 过滤掉在相机后方的点（z<=0 会投影炸掉）
        if np.any(corners_cam[:, 2] <= 0.1):
            continue

        corners_2d = project_to_image(corners_cam, P)  # 复用你已有函数

        cls_name = CLASSES[int(lb)] if 0 <= int(lb) < len(CLASSES) else "unk"
        color = COLOR.get(cls_name, (0,255,0))
        img = draw_box_3d(img, corners_2d, c=color)     # 复用你已有函数

        # 画文字（可选）
        u, v = int(corners_2d[:,0].mean()), int(corners_2d[:,1].mean())
        cv2.putText(img, f"{cls_name}:{sc:.2f}", (u, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return img


# 1) 读图
img = cv2.imread("data/dairv2x/image/010699.jpg")  # BGR

# 2) 读标定
K = load_K_from_camera_intrinsic_json("data/dairv2x/calib/camera_intrinsic/010699.json")
T_lidar_to_cam = load_T_lidar_to_cam_from_json("data/dairv2x/calib/virtuallidar_to_camera/010699.json")

scores = [
    0.716,
    0.695,
    0.609,
    0.410,
    0.390,
    0.341,
    0.330,
    0.302,
    0.297,
    0.259
]

labels = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
]

boxes = [
    [63.782711029052734, -3.4652328491210938, -1.7029876708984375, 2.0642731189727783, 4.410454750061035, 1.6725451946258545, 1.6620714664459229, -0.0035717152059078217, -0.0013145599514245987],
    [61.443294525146484, -10.262210845947266, -1.7129158973693848, 1.9648348093032837, 4.264991283416748, 1.5180312395095825, 1.3122766017913818, -0.0012612305581569672, 0.0018079746514558792],
    [18.17829704284668, 16.457195281982422, -1.1312096118927002, 1.6758549213409424, 4.130034923553467, 0.7850367426872253, 0.07823343575000763, -0.03118664212524891, -0.02437692880630493],
    [84.56058502197266, 25.020397186279297, -1.5679726600646973, 1.7244426012039185, 4.336910724639893, 1.5476946830749512, 3.1370654106140137, -0.0007921494543552399, -0.0001989658921957016],
    [12.102877616882324, 19.766101837158203, -1.5063824653625488, 1.7326687574386597, 4.593943119049072, 1.1644961833953857, 0.15784762799739838, -0.013779871165752411, 0.05828143656253815],
    [100.56697845458984, 25.005657196044922, -1.6559772491455078, 1.5741955041885376, 4.204702377319336, 1.3231369256973267, 3.1222052574157715, -0.00011062994599342346, 8.404813706874847e-05],
    [97.42134857177734, -22.237003326416016, -1.718440055847168, 1.7355053424835205, 4.2472124099731445, 1.3726139068603516, 0.7065179944038391, 0.004312139004468918, -0.001557283103466034],
    [77.42833709716797, 9.03314208984375, -1.6914877891540527, 1.870780348777771, 4.227057456970215, 1.387652039527893, 0.1660633236169815, 0.00017655640840530396, 0.0019294749945402145],
    [111.37487030029297, 25.02402114868164, -1.6388260126113892, 1.6678773164749146, 4.223330974578857, 1.3966273069381714, 3.0959174633026123, 0.000780981034040451, -0.0009198300540447235],
    [21.7126522064209, 19.339832305908203, -1.6155331134796143, 1.6839599609375, 4.19346284866333, 1.1740258932113647, -0.015200482681393623, -0.0019829683005809784, -0.009971404448151588]
]

# 3) 模型输出转 numpy
bboxes_np = np.array(boxes)
scores_np = np.array(scores)
labels_np = np.array(labels)

# 4) 可视化
vis = draw_centerpoint_preds_on_image(
    img, bboxes_np, scores_np, labels_np,
    K=K, T_lidar_to_cam=T_lidar_to_cam,
    score_thr=0.3,
)

cv2.imwrite("vis_010699.jpg", vis)
print("saved to vis_010699.jpg")