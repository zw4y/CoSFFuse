"""
SAM2 离线预处理：对数据集的 VI 图像自动生成实例分割 mask，存为 .pt 文件。

用法：
    python utils/preprocess_sam.py --dataset_root ./MSRS --mode train

前置准备：
    1. 安装 SAM2：
       pip install git+https://github.com/facebookresearch/sam2.git
       或：git clone https://github.com/facebookresearch/sam2 && pip install -e sam2

    2. 下载 SAM2 权重（以 large 为例）：
       wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
       放到项目根目录下的 checkpoints/ 文件夹中
       mkdir -p checkpoints
       mv sam2.1_hiera_large.pt checkpoints/

    3. 确认目录结构：
       checkpoints/
         └── sam2.1_hiera_large.pt

输出：
    在 {dataset_root}/{split}/sam_mask/ 下生成与 VI 图像同名的 .pt 文件，
    每个 .pt 是一个 (H, W) 的 LongTensor，像素值 ∈ {0, 1, 2, ..., N}，
    0=背景，其余为 instance ID。
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# ── SAM2 模型配置映射 ──────────────────────────────────────────────
# 注意：config_name 相对于 sam2 包的 config 路径，需加完整路径前缀
SAM2_CONFIGS = {
    "tiny":       "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small":      "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large":      "configs/sam2.1/sam2.1_hiera_l.yaml",
}

SAM2_CKPTS = {
    "tiny":       "sam2.1_hiera_tiny.pt",
    "small":      "sam2.1_hiera_small.pt",
    "base_plus":  "sam2.1_hiera_base_plus.pt",
    "large":      "sam2.1_hiera_large.pt",
}


def load_mask_generator(model_type: str = "large", device: str = "cuda"):
    """
    加载 SAM2 自动掩码生成器。
    返回 SAM2AutomaticMaskGenerator 实例。
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print("=" * 70)
        print("SAM2 未安装。请执行以下步骤：")
        print("  1. git clone https://github.com/facebookresearch/sam2")
        print("  2. cd sam2 && pip install -e .")
        print("  或: pip install git+https://github.com/facebookresearch/sam2.git")
        print("=" * 70)
        raise

    cfg_name = SAM2_CONFIGS.get(model_type)
    ckpt_name = SAM2_CKPTS.get(model_type)

    if cfg_name is None or ckpt_name is None:
        raise ValueError(f"未知 model_type: {model_type}，可选 {list(SAM2_CONFIGS.keys())}")

    # SAM2 内置的 yaml 配置文件在 sam2 包内，直接传文件名即可
    # checkpoint 路径需要在本地
    ckpt_path = os.path.join("checkpoints", ckpt_name)

    if not os.path.exists(ckpt_path):
        print("=" * 70)
        print(f"SAM2 权重文件未找到: {ckpt_path}")
        print(f"请下载 {ckpt_name} 放到 checkpoints/ 目录下。")
        print(f"下载地址: https://github.com/facebookresearch/sam2#download-checkpoints")
        print("=" * 70)
        raise FileNotFoundError(ckpt_path)

    print(f"加载 SAM2 ({model_type}) 模型...")
    sam2 = build_sam2(cfg_name, ckpt_path, device=device)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,          # 每边采样点数（平衡速度与精度）
        points_per_batch=64,         # 批量推理点数
        pred_iou_thresh=0.88,        # 预测 IoU 阈值
        stability_score_thresh=0.95, # 稳定性阈值
        stability_score_offset=1.0,
        box_nms_thresh=0.7,          # NMS 阈值
        crop_n_layers=0,             # 不裁剪（单尺度，加快速度）
        crop_nms_thresh=0.7,
        min_mask_region_area=100,    # 最小区域面积（过滤噪点）
    )
    return mask_generator


def generate_mask(mask_generator, image: np.ndarray) -> np.ndarray:
    """
    输入 RGB 图像 (H, W, 3) uint8，输出 instance mask (H, W) LongTensor。
    ID 从 1 开始编号（0 保留给背景/未分配区域）。
    大 mask 覆盖小 mask 的重叠区域时，保留大面积 mask 的 ID。
    """
    results = mask_generator.generate(image)

    if not results:
        # 无 mask → 全图一个区域
        h, w = image.shape[:2]
        return np.ones((h, w), dtype=np.int64)

    # 按面积降序排列，大面积优先
    results = sorted(results, key=lambda x: x["area"], reverse=True)

    h, w = image.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.int64)

    for i, r in enumerate(results):
        seg = r["segmentation"]  # bool (H, W)
        instance_mask[seg & (instance_mask == 0)] = i + 1

    return instance_mask


def process_split(dataset_root: str, split: str, model_type: str = "large"):
    """
    处理数据集的某个 split，生成 SAM mask 并存盘。
    支持断点续跑：已存在的 mask 跳过。
    """
    root = Path(dataset_root)
    vi_dir = root / split / "vi"
    out_dir = root / split / "sam_mask"

    if not vi_dir.exists():
        print(f"VI 目录不存在: {vi_dir}，跳过。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    img_names = sorted(os.listdir(vi_dir))
    # 收集已有 mask，支持断点续跑
    existing = {Path(f).stem for f in os.listdir(out_dir) if f.endswith(".pt")}
    pending = [n for n in img_names if Path(n).stem not in existing]

    if not pending:
        print(f"[{split}] 所有 mask 已存在，跳过（共 {len(img_names)} 张）。")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_generator = load_mask_generator(model_type, device)

    print(f"[{split}] 开始处理 {len(pending)} / {len(img_names)} 张图片...")

    for i, img_name in enumerate(tqdm(pending, desc=f"Processing {split}")):
        img_path = vi_dir / img_name
        image = np.array(Image.open(img_path).convert("RGB"))

        instance_mask = generate_mask(mask_generator, image)

        save_name = Path(img_name).stem + ".pt"
        torch.save(torch.from_numpy(instance_mask), out_dir / save_name)

        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    print(f"[{split}] 完成！{len(pending)} 张 mask 已保存至 {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM2 离线预处理：为数据集的 VI 图像生成 instance mask。"
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="数据集根目录（如 ./MSRS）")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "val", "all"],
                        help="处理哪个 split")
    parser.add_argument("--model_type", type=str, default="large",
                        choices=["tiny", "small", "base_plus", "large"],
                        help="SAM2 模型大小 (large 精度最高)")
    args = parser.parse_args()

    splits = {"train", "test", "val"} if args.mode == "all" else {args.mode}

    for split in splits:
        process_split(args.dataset_root, split, args.model_type)

    print("\n全部完成！可以开始训练了。")


if __name__ == "__main__":
    main()
