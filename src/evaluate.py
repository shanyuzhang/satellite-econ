"""
评估：测试集 R²/RMSE/MAE + VIIRS OLS 基线 + 误差与初始条件的相关性。

用法：
    python -m src.evaluate \
        --checkpoints M1:outputs/checkpoints/M1_level_best.pt \
                      M2:outputs/checkpoints/M2_level_best.pt \
        --mode level
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from .dataset import get_dataloaders
from .models import build_model
from .utils import get_device, load_config, mae, r2_score, rmse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="格式：M1:path M2:path ...")
    ap.add_argument("--mode", default="level", choices=["level", "diff1"])
    ap.add_argument("--out", default="outputs/figures/results_table.csv")
    return ap.parse_args()


def predict_model(model, loader, device):
    model.eval()
    preds, labels, grid_ids, years = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            feat = batch["features"].to(device, non_blocking=True)
            y = batch["label"].numpy()
            p = model(img, feat).squeeze(-1).cpu().numpy()
            preds.append(p)
            labels.append(y)
            grid_ids.extend(batch["grid_id"])
            years.extend([int(x) for x in batch["year"]])
    return (np.concatenate(preds), np.concatenate(labels),
            np.array(grid_ids), np.array(years))


def viirs_ols_baseline(config: dict, mode: str):
    """用 train 集拟合 (viirs_z, base_log_gdp_z) → 目标的 OLS，返回 test 集预测。"""
    labels = pd.read_csv(config["data"]["paths"]["labels_csv"])
    target_col = "log_gdp" if mode == "level" else "diff_log_gdp"
    feats = config["data"]["numeric_features"]

    # 去掉目标列和特征列里任何 NaN（边缘网格 VIIRS 缺测会有）
    labels = labels.dropna(subset=[target_col] + feats)

    train_df = labels[labels["split"] == "train"]
    test_df = labels[labels["split"] == "test"]

    reg = LinearRegression().fit(train_df[feats].values, train_df[target_col].values)
    y_pred = reg.predict(test_df[feats].values)
    y_true = test_df[target_col].values
    return y_pred, y_true, test_df


def main():
    args = parse_args()
    config = load_config(args.config)
    config["train"]["mode"] = args.mode

    device = get_device()

    loaders = get_dataloaders(config)
    num_features = loaders["num_features"]

    # 评估各模型
    results = []
    test_preds = {}  # for visualize
    for spec in args.checkpoints:
        if ":" not in spec:
            raise ValueError(f"checkpoints 格式错误：{spec}（应为 M1:path）")
        name, path = spec.split(":", 1)
        print(f"\n[评估] {name} from {path}")
        ckpt = torch.load(path, map_location="cpu")
        model = build_model(name, num_features=num_features, config=config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        y_pred, y_true, grid_ids, years = predict_model(model, loaders["test"], device)
        r2 = r2_score(y_true, y_pred)
        rm = rmse(y_true, y_pred)
        ma = mae(y_true, y_pred)
        results.append({"model": name, "mode": args.mode,
                        "n_test": len(y_true), "R2": r2, "RMSE": rm, "MAE": ma})
        test_preds[name] = {"pred": y_pred, "true": y_true,
                            "grid_id": grid_ids, "year": years}
        print(f"  R²={r2:.4f}  RMSE={rm:.4f}  MAE={ma:.4f}  n={len(y_true)}")

    # VIIRS OLS 基线
    print(f"\n[评估] VIIRS+base_log_gdp OLS 基线")
    y_pred_ols, y_true_ols, ols_df = viirs_ols_baseline(config, args.mode)
    r2 = r2_score(y_true_ols, y_pred_ols)
    rm = rmse(y_true_ols, y_pred_ols)
    ma = mae(y_true_ols, y_pred_ols)
    results.append({"model": "OLS(VIIRS+base)", "mode": args.mode,
                    "n_test": len(y_true_ols), "R2": r2, "RMSE": rm, "MAE": ma})
    print(f"  R²={r2:.4f}  RMSE={rm:.4f}  MAE={ma:.4f}  n={len(y_true_ols)}")

    # 误差与初始条件的相关性（仅对深度模型，检查系统偏差）
    print(f"\n[诊断] 残差 vs base_log_gdp 的 Pearson 相关：")
    labels_full = pd.read_csv(config["data"]["paths"]["labels_csv"])
    for name, d in test_preds.items():
        resid = d["pred"] - d["true"]
        # 把每个样本对齐回 base_log_gdp
        merged = pd.DataFrame({
            "grid_id": d["grid_id"], "year": d["year"], "resid": resid,
        }).merge(
            labels_full[["grid_id", "year", "base_log_gdp"]],
            on=["grid_id", "year"], how="left",
        )
        merged = merged.dropna(subset=["base_log_gdp"])
        if len(merged) > 1:
            corr = np.corrcoef(merged["resid"], merged["base_log_gdp"])[0, 1]
            print(f"  {name}: corr(residual, base_log_gdp) = {corr:.4f}")
        else:
            print(f"  {name}: 样本不足，跳过")

    # 写出结果表
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    print(f"\n[输出] 结果表：{out_path}")
    print("\nMarkdown 摘要：")
    print(res_df.to_markdown(index=False, floatfmt=".4f"))

    # 顺便保存预测结果，给 visualize 用
    pred_dump = []
    for name, d in test_preds.items():
        for i in range(len(d["pred"])):
            pred_dump.append({
                "model": name, "grid_id": d["grid_id"][i], "year": d["year"][i],
                "y_true": float(d["true"][i]), "y_pred": float(d["pred"][i]),
            })
    pred_path = out_path.parent / f"predictions_{args.mode}.csv"
    pd.DataFrame(pred_dump).to_csv(pred_path, index=False)
    print(f"[输出] 模型预测明细：{pred_path}")


if __name__ == "__main__":
    main()
