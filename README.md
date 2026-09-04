# 🌟 RECO

> RECO is a robust roadside 3D perception framework designed for camera-based BEV / 3D object detection under **uncertain camera extrinsics**. 

## ✨ About The Project

* 🚀 **Robust to extrinsic perturbations**: RECO improves detection stability when camera extrinsics are noisy, drifting, or slightly mismatched in real roadside environments.

* 🏗️ **Easy to integrate into BEV pipelines**: RECO is designed around projection-based BEV / 3D detection frameworks and can be incorporated into existing camera-only pipelines.

## 🛠️ Getting Started
Please follow the guides below to set up and run the project:

- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Evaluation](docs/run_and_eval.md)

## 🧪 single test

You can run a single-sample test for quick case.

Example yaw (0,0.5):

set perturbation condiction in the  dataset/nusc_mv_dataset.py

> mu_yaw_deg = 0 
> sigma_yaw_deg = 0.5 

run bash

```bash

python exps/dair-v2x/exps_reco.py --ckpt_path ckpts/z_yaw_pitch_roll.ckpt -e -b 32 --gpus 1
```

## 📊 Detection Results

We report the results under under z-yaw-pitch-roll N(0,0.5) perturbations.

### Car (IoU = 0.5)

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| BBox AP | 73.8036 | 67.5365 | 67.6009 |
| BEV AP  | 82.3296 | 76.6615 | 76.7015 |
| 3D AP   | 71.9712 | 69.9732 | 70.0426 |
| AOS     | 73.66 | 67.39 | 67.45 |

### Pedestrian (IoU = 0.25)

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| BBox AP | 40.2271 | 39.3415 | 39.8315 |
| BEV AP  | 17.7391 | 16.7309 | 16.9414 |
| 3D AP   | 16.2505 | 15.2773 | 15.4720 |
| AOS     | 37.66 | 36.75 | 37.24 |

### Cyclist (IoU = 0.25)

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| BBox AP | 71.1393 | 73.0900 | 73.5359 |
| BEV AP  | 45.3320 | 48.8390 | 49.5848 |
| 3D AP   | 47.8113 | 45.8919 | 42.5776 |
| AOS     | 69.49 | 71.56 | 71.97 |


## 🤝 Contributing
This project is built upon several excellent open-source works in camera-based BEV perception and roadside 3D detection.

We would like to sincerely thank the authors and contributors of:

* BEVHeight
* CoBEV
* BEVSpread
* HeightFormer

