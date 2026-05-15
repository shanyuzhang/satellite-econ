"""
可视化：
  - 预测 vs 实际散点图（每个模型一张）
  - 所有模型 R² 对比柱状图
  - M2 的 Transformer CLS→patch 注意力热力图

用法：
    python -m src.visualize \
        --results outputs/figures/results_table.csv \
        --predictions outputs/figures/predictions_level.csv

    # M2 注意力热力图（需要训练好的 M2 checkpoint）
    python -m src.visualize attention \
        --checkpoint outputs/checkpoints/M2_level_best.pt \
        --n-samples 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .dataset import get_dataloaders, read_tif, resize_chw
from .models import build_model
from .utils import get_device, load_config, r2_score


def plot_scatter(predictions_csv: str, out_dir: str):
    df = pd.read_csv(predictions_csv)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for model_name, grp in df.groupby("model"):
        r2 = r2_score(grp["y_true"].values, grp["y_pred"].values)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(grp["y_true"], grp["y_pred"], s=6, alpha=0.4)
        lo = min(grp["y_true"].min(), grp["y_pred"].min())
        hi = max(grp["y_true"].max(), grp["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.set_title(f"{model_name}   R²={r2:.4f}   n={len(grp)}")
        ax.set_aspect("equal")
        out = Path(out_dir) / f"scatter_{model_name}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[scatter] {out}")


def plot_r2_bar(results_csv: str, out_path: str):
    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    bars = ax.bar(x, df["R2"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].values, rotation=20, ha="right")
    ax.set_ylabel("Test R²")
    mode = df["mode"].iloc[0] if "mode" in df.columns else ""
    ax.set_title(f"模型对比（{mode}）")
    for i, v in enumerate(df["R2"].values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[bar] {out_path}")


def plot_attention(checkpoint: str, config_path: str, n_samples: int, out_dir: str):
    config = load_config(config_path)
    device = get_device()

    ckpt = torch.load(checkpoint, map_location="cpu")
    config["train"]["mode"] = ckpt["mode"]
    loaders = get_dataloaders(config)
    num_features = loaders["num_features"]

    model = build_model("M2", num_features=num_features, config=config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    shown = 0
    for batch in loaders["test"]:
        img = batch["image"].to(device)
        feat = batch["features"].to(device)
        with torch.no_grad():
            pred, attn = model.forward_with_attention(img, feat)
        # attn: (B, 14, 14)
        for i in range(img.shape[0]):
            if shown >= n_samples:
                return
            grid_id = batch["grid_id"][i]
            year = int(batch["year"][i])
            y = float(batch["label"][i])
            yh = float(pred[i].item())

            # 重新从原文件读 RGB（B4/B3/B2）做底图（已 z-score 的 batch 不好看）
            img_dir = config["data"]["paths"]["sentinel2_dir"]
            n_bands = len(config["data"]["bands"])
            size = config["data"]["image_size"]
            tif = Path(img_dir) / f"{grid_id}_{year}.tif"
            if tif.exists():
                arr = read_tif(str(tif), n_bands)
                arr = resize_chw(arr, size)
                # B4=index 2, B3=1, B2=0（按 config 顺序）
                rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
                # 简单百分位拉伸
                lo, hi = np.percentile(rgb, [2, 98])
                rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
            else:
                rgb = np.zeros((size, size, 3))

            attn_map = attn[i].cpu().numpy()
            # 上采样到 224
            attn_t = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
            attn_up = F.interpolate(attn_t, size=(size, size),
                                     mode="bilinear", align_corners=False).squeeze().numpy()
            attn_up = (attn_up - attn_up.min()) / max(attn_up.max() - attn_up.min(), 1e-6)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb)
            axes[0].set_title(f"{grid_id} {year}\ny={y:.3f}  ŷ={yh:.3f}")
            axes[0].axis("off")
            axes[1].imshow(rgb)
            axes[1].imshow(attn_up, cmap="hot", alpha=0.5)
            axes[1].set_title("CLS→patch attention")
            axes[1].axis("off")
            out = Path(out_dir) / f"attn_{grid_id}_{year}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"[attn] {out}")
            shown += 1
        if shown >= n_samples:
            return


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p_summary = sub.add_parser("summary", help="散点图 + R² 柱状图")
    p_summary.add_argument("--results", required=True)
    p_summary.add_argument("--predictions", required=True)
    p_summary.add_argument("--out-dir", default="outputs/figures")

    p_attn = sub.add_parser("attention", help="M2 注意力热力图")
    p_attn.add_argument("--checkpoint", required=True)
    p_attn.add_argument("--config", default="configs/default.yaml")
    p_attn.add_argument("--n-samples", type=int, default=6)
    p_attn.add_argument("--out-dir", default="outputs/figures/attention")

    args = ap.parse_args()

    if args.cmd == "summary":
        plot_scatter(args.predictions, args.out_dir)
        plot_r2_bar(args.results, str(Path(args.out_dir) / "r2_bar.png"))
    elif args.cmd == "attention":
        plot_attention(args.checkpoint, args.config, args.n_samples, args.out_dir)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
