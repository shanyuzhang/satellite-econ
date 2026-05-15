"""
02_prepare_labels.py — 把县级 GDP、网格元数据、VIIRS 合并成训练用的 labels.csv，并按县划分 train/valid/test。

输入：
  - data/raw/guangdong_county_gdp.csv （用户提供；格式见 guangdong_county_gdp_template.csv）
  - data/processed/grid_meta.csv      （由 01_export_gee.py 生成；含 grid_id / lat / lng / county_code / county_name）
  - data/processed/viirs.csv          （由 01_export_gee.py 生成；含 grid_id / year / viirs_mean）

输出：
  - data/processed/labels.csv         （含 grid_id, year, log_gdp, diff_log_gdp, base_log_gdp, viirs_mean, *_z, split）
  - data/processed/split.json         （{train: [grid_ids], valid: [...], test: [...]}）
  - data/processed/feature_stats.json （训练集数值特征 mean/std，用于推断时复用）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_gdp_to_grids(gdp_df: pd.DataFrame, grid_meta: pd.DataFrame) -> pd.DataFrame:
    """
    简化方案：县 GDP 在该县所有 grid 上均匀分配。
    grid_meta: grid_id, county_name (来自 GAUL ADM2_NAME), ...
    gdp_df: county_name, year, gdp_billion_yuan
    返回：grid_id, year, county_name, gdp_billion_yuan_grid

    用 county_name 而非 county_code join，因为 GAUL ADM2_CODE 与中国国统局 6 位县码是
    两套独立编码系统。
    """
    # 标准化 county_name：strip + 转小写
    gdp_df = gdp_df.copy()
    grid_meta = grid_meta.copy()
    gdp_df["_cn"] = gdp_df["county_name"].astype(str).str.strip().str.lower()
    grid_meta["_cn"] = grid_meta["county_name"].astype(str).str.strip().str.lower()

    # 每个县的 grid 数
    grids_per_county = grid_meta.groupby("_cn").size().rename("n_grids").reset_index()
    merged = gdp_df.merge(grids_per_county, on="_cn", how="inner")
    if len(merged) == 0:
        gnames = sorted(set(gdp_df["_cn"]))
        mnames = sorted(set(grid_meta["_cn"]))
        raise RuntimeError(
            f"gdp_df 和 grid_meta 的 county_name 没交集。\n"
            f"gdp 里有: {gnames}\n"
            f"grid_meta 里有: {mnames}"
        )
    merged["gdp_per_grid"] = merged["gdp_billion_yuan"] / merged["n_grids"]

    # 展开到每个 grid
    g2c = grid_meta[["grid_id", "_cn", "county_name"]]
    out = g2c.merge(merged[["_cn", "year", "gdp_per_grid"]], on="_cn", how="inner")
    out = out.rename(columns={"gdp_per_grid": "gdp_billion_yuan_grid"}) \
             .drop(columns=["_cn"])
    return out[["grid_id", "year", "county_name", "gdp_billion_yuan_grid"]]


def split_by_county(county_codes, train_r, valid_r, test_r, seed):
    """按 county_code 三分组划分。返回 {county_code: split}。"""
    rng = np.random.RandomState(seed)
    counties = sorted(set(county_codes))
    rng.shuffle(counties)
    n = len(counties)
    n_train = int(n * train_r)
    n_valid = int(n * valid_r)
    mapping = {}
    for i, c in enumerate(counties):
        if i < n_train:
            mapping[c] = "train"
        elif i < n_train + n_valid:
            mapping[c] = "valid"
        else:
            mapping[c] = "test"
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["data"]["paths"]
    base_year = cfg["data"]["base_year"]

    # 1. 读输入
    gdp_path = Path(paths["gdp_csv"])
    if not gdp_path.exists():
        print(f"[错误] 未找到 {gdp_path}。请参照 {paths['gdp_template_csv']} 的格式准备。")
        sys.exit(1)

    gdp_df = pd.read_csv(gdp_path)
    grid_meta = pd.read_csv(paths["grid_meta_csv"])
    viirs = pd.read_csv(paths["viirs_csv"])

    print(f"[输入] gdp_df={len(gdp_df)} rows, grid_meta={len(grid_meta)} grids, viirs={len(viirs)} rows")

    # 2. 县 GDP → 网格 GDP（均匀分配，按 county_name join）
    grid_gdp = assign_gdp_to_grids(gdp_df, grid_meta)
    print(f"[标签] 网格-年级别 GDP 行数：{len(grid_gdp)}")

    # 3. log GDP 和年度差分
    grid_gdp = grid_gdp.sort_values(["grid_id", "year"]).reset_index(drop=True)
    grid_gdp["log_gdp"] = np.log(grid_gdp["gdp_billion_yuan_grid"])
    grid_gdp["diff_log_gdp"] = grid_gdp.groupby("grid_id")["log_gdp"].diff()

    # 4. base_log_gdp = 基期（base_year）的县 log GDP（用县级 GDP 而非分摊后的 grid GDP）
    gdp_df["log_gdp_county"] = np.log(gdp_df["gdp_billion_yuan"])
    base = gdp_df[gdp_df["year"] == base_year][["county_name", "log_gdp_county"]] \
        .rename(columns={"log_gdp_county": "base_log_gdp"})
    if len(base) == 0:
        print(f"[警告] 基期 {base_year} 在 gdp 数据中没有任何记录，base_log_gdp 将全 NaN。")
    grid_gdp = grid_gdp.merge(base, on="county_name", how="left")

    # 5. VIIRS 合并（每个 grid×year 一个值）
    df = grid_gdp.merge(viirs, on=["grid_id", "year"], how="left")

    # 6. 划分（按 county_name 防泄漏；同一地级市的网格-年全在同一 split）
    split_map = split_by_county(
        df["county_name"].unique(),
        cfg["data"]["split"]["train_ratio"],
        cfg["data"]["split"]["valid_ratio"],
        cfg["data"]["split"]["test_ratio"],
        cfg["data"]["split"]["seed"],
    )
    df["split"] = df["county_name"].map(split_map)

    # 7. 用训练集统计量做 z-score
    #    数值特征列名映射：源列 → 输出 z-score 列
    z_cols = {
        "base_log_gdp": "base_log_gdp_z",
        "viirs_mean":   "viirs_z",
    }
    train_mask = df["split"] == "train"
    stats = {}
    for src, dst in z_cols.items():
        mean = df.loc[train_mask, src].mean()
        std = df.loc[train_mask, src].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        stats[src] = {"mean": float(mean), "std": float(std)}
        df[dst] = (df[src] - mean) / std

    # 8. 写出
    processed_dir = Path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(paths["labels_csv"], index=False)
    print(f"[输出] labels: {paths['labels_csv']} ({len(df)} rows)")

    split_dict = {
        "train": sorted(df.loc[df["split"] == "train", "grid_id"].unique().tolist()),
        "valid": sorted(df.loc[df["split"] == "valid", "grid_id"].unique().tolist()),
        "test":  sorted(df.loc[df["split"] == "test",  "grid_id"].unique().tolist()),
    }
    with open(paths["split_json"], "w") as f:
        json.dump(split_dict, f, indent=2)
    print(f"[输出] split: {paths['split_json']} "
          f"(train={len(split_dict['train'])}, valid={len(split_dict['valid'])}, test={len(split_dict['test'])})")

    with open(paths["feature_stats_json"], "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[输出] feature_stats: {paths['feature_stats_json']}")

    # 简要检查
    print("\n[检查] 各 split 的 (grid, year) 数量：")
    print(df.groupby("split").size())
    print("\n[检查] county 是否跨 split 泄漏（应全部 == 1）：")
    leak = df.groupby("county_name")["split"].nunique()
    print(f"  county_name → split nunique 最大值 = {leak.max()}")


if __name__ == "__main__":
    main()
