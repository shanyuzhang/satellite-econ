"""
PyTorch Dataset：广东省网格 Sentinel-2 + 数值特征 → log GDP。

文件命名约定：data/sentinel2/{grid_id}_{year}.tif，9 波段顺序与 configs.data.bands 一致。

标签模式：
  - level: log_gdp
  - diff1: diff_log_gdp（年度差分；t-1 缺失的样本自动剔除）
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import load_config, load_json, save_json


# -----------------------
# 数据增强（对多通道 ndarray 操作）
# -----------------------
def random_augment(img: np.ndarray, cfg: dict) -> np.ndarray:
    """img: (C, H, W) numpy。原地翻转和 90° 旋转。"""
    if cfg.get("hflip", False) and random.random() < 0.5:
        img = img[:, :, ::-1].copy()
    if cfg.get("vflip", False) and random.random() < 0.5:
        img = img[:, ::-1, :].copy()
    if cfg.get("rot90", False):
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
    return img


# -----------------------
# 图像加载和缩放
# -----------------------
def read_tif(path: str, n_bands: int) -> np.ndarray:
    """读 GeoTIFF，返回 (C, H, W) float32，缺失波段补 0。"""
    with rasterio.open(path) as src:
        arr = src.read(out_dtype="float32")  # (bands, H, W)
    if arr.shape[0] < n_bands:
        pad = np.zeros((n_bands - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > n_bands:
        arr = arr[:n_bands]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def resize_chw(arr: np.ndarray, size: int) -> np.ndarray:
    """简单双线性 resize 到 size×size。"""
    import torch.nn.functional as F
    t = torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


# -----------------------
# 训练集逐波段统计量
# -----------------------
def compute_band_stats(df_train: pd.DataFrame, image_dir: str, n_bands: int,
                       image_size: int, max_samples: int = 500) -> dict:
    """随机采样最多 max_samples 张训练图，估算逐波段 mean/std。"""
    sample_rows = df_train.sample(n=min(max_samples, len(df_train)), random_state=42)
    sums = np.zeros(n_bands, dtype=np.float64)
    sumsq = np.zeros(n_bands, dtype=np.float64)
    pixels = 0
    missing = 0

    for _, row in tqdm(sample_rows.iterrows(), total=len(sample_rows),
                       desc="计算波段统计量"):
        p = Path(image_dir) / f"{row['grid_id']}_{int(row['year'])}.tif"
        if not p.exists():
            missing += 1
            continue
        arr = read_tif(str(p), n_bands)
        arr = resize_chw(arr, image_size)
        c = arr.shape[0]
        n = arr.shape[1] * arr.shape[2]
        sums += arr.reshape(c, -1).sum(axis=1)
        sumsq += (arr.reshape(c, -1) ** 2).sum(axis=1)
        pixels += n

    if pixels == 0:
        raise RuntimeError(f"没有任何可读图像（缺失 {missing}）。请检查 image_dir 是否正确。")

    mean = sums / pixels
    var = sumsq / pixels - mean ** 2
    var = np.clip(var, 1e-8, None)
    std = np.sqrt(var)
    return {"mean": mean.tolist(), "std": std.tolist(), "n_samples": int(len(sample_rows) - missing)}


# -----------------------
# Dataset
# -----------------------
class GuangdongGridDataset(Dataset):
    def __init__(
        self,
        labels_df: pd.DataFrame,
        image_dir: str,
        bands: list,
        image_size: int,
        numeric_features: list,
        band_stats: dict,
        target_col: str,
        augment_cfg: Optional[dict] = None,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.n_bands = len(bands)
        self.image_size = image_size
        self.numeric_features = numeric_features
        self.target_col = target_col
        self.band_mean = np.array(band_stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.band_std = np.array(band_stats["std"], dtype=np.float32).reshape(-1, 1, 1)
        self.augment_cfg = augment_cfg  # None 表示不增强

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        grid_id = row["grid_id"]
        year = int(row["year"])

        img_path = self.image_dir / f"{grid_id}_{year}.tif"
        if not img_path.exists():
            # 缺失图像：返回全 0（也可改为抛错；这里选择鲁棒）
            img = np.zeros((self.n_bands, self.image_size, self.image_size), dtype=np.float32)
        else:
            img = read_tif(str(img_path), self.n_bands)
            img = resize_chw(img, self.image_size)

        # 标准化
        img = (img - self.band_mean) / self.band_std

        # 增强
        if self.augment_cfg is not None:
            img = random_augment(img, self.augment_cfg)

        # 数值特征
        feats = np.array([row[c] for c in self.numeric_features], dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        label = float(row[self.target_col])

        return {
            "image": torch.from_numpy(img.copy()).float(),
            "features": torch.from_numpy(feats).float(),
            "label": torch.tensor(label, dtype=torch.float32),
            "grid_id": grid_id,
            "year": year,
        }


# -----------------------
# DataLoader 工厂
# -----------------------
def _filter_labels_for_mode(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    """返回过滤后的 df 和目标列名。"""
    if mode == "level":
        df = df.dropna(subset=["log_gdp"]).copy()
        return df, "log_gdp"
    if mode == "diff1":
        df = df.dropna(subset=["diff_log_gdp"]).copy()
        return df, "diff_log_gdp"
    raise ValueError(f"未知 mode: {mode}")


def get_dataloaders(config: dict):
    """
    根据配置返回 (train, valid, test) DataLoader 以及 num_features。
    若 band_stats.json 不存在则计算并缓存。
    """
    paths = config["data"]["paths"]
    bands = config["data"]["bands"]
    image_size = config["data"]["image_size"]
    numeric_features = config["data"]["numeric_features"]

    df = pd.read_csv(paths["labels_csv"])
    df, target_col = _filter_labels_for_mode(df, config["train"]["mode"])

    df_train = df[df["split"] == "train"].copy()
    df_valid = df[df["split"] == "valid"].copy()
    df_test  = df[df["split"] == "test"].copy()

    # 波段统计量（缓存到 band_stats.json）
    stats_path = paths["band_stats_json"]
    if not Path(stats_path).exists():
        print(f"[Dataset] 未找到 {stats_path}，开始计算训练集逐波段统计量...")
        stats = compute_band_stats(
            df_train,
            image_dir=paths["sentinel2_dir"],
            n_bands=len(bands),
            image_size=image_size,
        )
        save_json(stats, stats_path)
        print(f"[Dataset] 已写入 {stats_path}")
    band_stats = load_json(stats_path)

    aug_cfg = config["train"]["augment"]

    train_ds = GuangdongGridDataset(
        df_train, paths["sentinel2_dir"], bands, image_size,
        numeric_features, band_stats, target_col, augment_cfg=aug_cfg,
    )
    valid_ds = GuangdongGridDataset(
        df_valid, paths["sentinel2_dir"], bands, image_size,
        numeric_features, band_stats, target_col, augment_cfg=None,
    )
    test_ds = GuangdongGridDataset(
        df_test, paths["sentinel2_dir"], bands, image_size,
        numeric_features, band_stats, target_col, augment_cfg=None,
    )

    bs = config["train"]["batch_size"]
    nw = config["train"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test":  test_loader,
        "num_features": len(numeric_features),
        "target_col": target_col,
    }
