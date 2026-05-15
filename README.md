# 卫星图像预测广东省经济活动 — 探索 ViT+CNN 混合架构

本项目是一个**探索性实验**，验证 ResNet→Transformer 混合架构在广东省 2.4km 网格级 GDP 预测上的可行性。
方法论参考 Khachiyan et al. (2022, AER Insights) 和 Cheng & Jiang (2024)。

## 模型

| 代号 | 架构 | 优先级 | 说明 |
|------|------|--------|------|
| M1 | ResNet-50 (9 通道) + AFF | 优先跑 | 纯 CNN 基线 |
| M2 | ResNet-50 前3 stage → Transformer + AFF | **主模型** | CNN 抽局部特征，Transformer 建全局关系 |
| M3 | ViT-Base (timm) + AFF | 次优先 | 纯 ViT，对比"提升来自 Transformer 还是混合设计" |
| M4 | SatMAE / ViT-Large 微调 + AFF | 后续探索 | 遥感预训练 vs ImageNet 预训练 |

预测目标：每个 2.4km 网格的 log GDP（level 模式）或年度差分 log GDP（diff1 模式）。

## 安装

```bash
# 建议 Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# GEE 认证（首次）
earthengine authenticate
```

## 数据准备流程

### 1. 用 GEE 导出 Sentinel-2 + VIIRS

```bash
# 冒烟测试（不真正提交任务）
python scripts/01_export_gee.py --years 2015 --max-tasks 5 --dry-run

# 真正提交（异步导出到 Google Drive 的 gee_guangdong_s2/ 文件夹）
python scripts/01_export_gee.py --years 2015-2023
```

任务异步执行，脚本会每 60s 打印状态。完成后从 Google Drive 下载所有 GeoTIFF 到 `data/sentinel2/`，
文件名格式 `{grid_id}_{year}.tif`。同时会生成 `data/processed/grid_meta.csv` 和 `data/processed/viirs.csv`。

### 2. 准备 GDP 标签

将你的县级 GDP 数据放到 `data/raw/guangdong_county_gdp.csv`，格式参考
`data/raw/guangdong_county_gdp_template.csv`：

```csv
county_code,county_name,year,gdp_billion_yuan
440103,广州市荔湾区,2015,800.5
...
```

然后运行：

```bash
python scripts/02_prepare_labels.py --config configs/default.yaml
```

输出 `data/processed/labels.csv`、`split.json`、`feature_stats.json`。

## 训练

```bash
# wandb 登录（首次）
wandb login

# M1 基线，level 模式
python -m src.train --model M1 --mode level

# M2 主模型，level + diff1
python -m src.train --model M2 --mode level
python -m src.train --model M2 --mode diff1

# M3/M4（次优先，代码已生成）
python -m src.train --model M3 --mode level
python -m src.train --model M4 --mode level   # 需先下载 SatMAE 权重，否则回退到 ImageNet
```

最佳权重保存到 `outputs/checkpoints/{model}_{mode}_best.pt`。

## 评估和可视化

```bash
python -m src.evaluate \
    --checkpoints M1:outputs/checkpoints/M1_level_best.pt \
                  M2:outputs/checkpoints/M2_level_best.pt \
    --mode level

python -m src.visualize --results outputs/figures/results_table.csv
```

## SatMAE 权重（M4 可选）

从 [sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE) 下载 Sentinel-2 预训练的
ViT-Large checkpoint，放到 `weights/satmae_vit_large.pth`。详见 `weights/README.md`。
如果权重不存在，M4 会自动回退到 timm 的 ImageNet ViT-Large 预训练。

## 项目结构

```
configs/default.yaml          # 集中超参数
scripts/01_export_gee.py      # GEE 导出
scripts/02_prepare_labels.py  # 标签 + 数值特征 + 划分
src/dataset.py                # PyTorch Dataset
src/models/                   # AFF + M1/M2/M3/M4
src/train.py                  # 训练入口
src/evaluate.py               # 测试集评估 + VIIRS OLS 基线
src/visualize.py              # 散点图、柱状图、注意力热力图
data/                         # 原始 CSV / GeoTIFF / 处理后产物
outputs/                      # checkpoints、日志、图
```

## 配置 wandb

在 `configs/default.yaml` 里填入你的 `wandb.entity`（团队/用户名），或者把 `wandb.mode` 设为
`offline`（本地缓存）或 `disabled`（关闭）。
