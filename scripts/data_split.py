import json
import random
from pathlib import Path
import argparse

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="图片目录，例如 ./image")
    parser.add_argument("--out", type=str, default="split.json", help="输出json文件名")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test 比例之和必须为 1.0，但当前是 {ratios_sum}")

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"目录不存在: {img_dir}")

    # 收集所有图片文件，取文件名去掉后缀作为 id
    ids = []
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            ids.append(p.stem)  # e.g. "010699"

    if not ids:
        raise RuntimeError(f"在 {img_dir} 下没有找到图片文件")

    random.seed(args.seed)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    # 剩下的全给 test，避免四舍五入导致总数不一致
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    split = {"train": train_ids, "val": val_ids, "test": test_ids}

    # 保存
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    # 同时打印到屏幕（就是你要的 {"train": [...]} 格式）
    print(json.dumps(split, ensure_ascii=False, indent=2))
    print(f"\nDone. total={n}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    main()
