# 项目：探索 ViT+CNN 混合架构预测广东省经济活动

## 项目性质

这是一个**探索性实验项目**，目的是验证 ViT+CNN 混合架构在中国卫星经济预测任务上的可行性。不是一个完整的论文实验——暂不追求全面的消融实验和基线对比，而是先把核心管线跑通、拿到初步结果、看看效果如何，再决定后续方向。

## 参照论文

本项目的方法论综合参考两篇论文：

**Khachiyan et al. (2022, AER: Insights)**："Using Neural Networks to Predict Microspatial Economic Growth"
- 纯 CNN 预测美国 2.4km 网格级收入和人口，R² 水平值 0.85-0.91，十年差分 0.32-0.46
- 关键设计：年度合成图像（5-8月中位数）、人口权重插值标签、初始条件控制变量、按城市区域划分数据集防泄漏

**Cheng & Jiang (2024)**："A View Across the Skyline: Nowcasting Combining Satellite and Energy Indicators"
- ResNet-50 + AFF（注意力特征融合）融合图像与能源数据，加州 58 县，R² 水平值 0.97
- 关键设计：AFF 比简单拼接提升显著、多频率差分（年度/五年/十年）

## 我要尝试的事情

在上述两篇论文的基础上，尝试一个改进：**用 CNN→Transformer 串行混合架构替代纯 CNN**，看看 Transformer 的全局空间建模能力对经济预测是否有增量价值。用 Sentinel-2 图像，在广东省的数据上跑一下看效果。

具体来说：
- ResNet-50 前三层提取局部特征 → Transformer 编码器建模全局空间关系 → 预测 GDP
- 用 AFF 注意力机制（借鉴 Cheng & Jiang）融合图像特征与数值特征
- 和纯 ResNet-50 基线做对比，看 R² 有没有提升
- 同时生成好纯 ViT 基线和 SatMAE 微调的代码，方便后续对比使用

## 任务

请帮我构建这个实验项目的完整代码。由于是探索阶段，优先保证**管线能跑通、代码结构清晰、后续容易扩展**，不需要追求极致优化。

---

### 模块一：GEE 数据导出

写 `scripts/01_export_gee.py`（geemap + earthengine-api）：

- 加载广东省行政边界（GEE 内置的 FAO GAUL 或 GADM 数据集）
- 将广东省划分为 2.4km × 2.4km 网格
- 用 GHSL 或类似数据过滤掉纯山地/森林/水域网格，只保留城镇化区域
- 对每个网格、每年（2015-2023），从 `COPERNICUS/S2_SR_HARMONIZED` 提取 5-8 月无云中位数合成
- 云掩膜用 SCL 波段，排除云和云阴影
- 波段选择：B2, B3, B4, B5, B6, B7, B8, B11, B12（共 9 波段）
- 导出 GeoTIFF，scale=10
- 同时导出 VIIRS 夜间灯光年度均值（CSV，用于基线对比和作为数值特征）
- 同时导出网格元数据 CSV：grid_id, center_lat, center_lng, county_code, county_name
- 支持批量提交 Export 任务，有进度监控

### 模块二：标签和特征处理

写 `scripts/02_prepare_labels.py`：

- 输入：用户自备的 `data/raw/guangdong_county_gdp.csv`（county_code, year, gdp_billion_yuan）
- 请提供一个模板 CSV 文件说明格式
- 将县级 GDP 按网格数量均匀分配到该县的各网格（简化方案，先不做人口权重插值）
- 计算 log GDP 和年度差分 diff_log_gdp
- 准备数值特征：基期（2015年）县级 log GDP、VIIRS 灯光均值，z-score 标准化
- 按县划分 train/valid/test（60%/20%/20%），同县网格同组，seed=42
- 输出处理好的 CSV 和 split.json

### 模块三：PyTorch Dataset

写 `src/dataset.py`：

- 读 GeoTIFF，resize 到 224×224，逐波段 z-score 标准化（训练集统计量）
- 训练集做随机翻转和旋转增强
- 返回 `{"image": tensor, "features": tensor, "label": scalar, "grid_id": str}`
- 支持 level 和 diff1 两种标签模式
- 提供 `get_dataloaders(config)` 函数

### 模块四：模型（核心）

统一接口：image (B, 9, 224, 224) + features (B, num_features) → prediction (B, 1)

**AFF 模块** (`src/models/aff.py`)：
- 借鉴 Cheng & Jiang，实现注意力特征融合
- 输入两个特征向量（维度可不同，内部先投影到相同维度）
- 局部注意力 + 全局注意力 → Sigmoid 权重 → 加权融合
- 所有模型共用

**M1: ResNet-50 基线** (`src/models/resnet_baseline.py`)：
- 9 通道输入，ImageNet 预训练（前 3 通道复用 RGB 权重）
- 全局平均池化 → 2048 维 → AFF 融合数值特征 → FC → 预测值

**M2: CNN→Transformer 混合（主模型）** (`src/models/cnn_transformer.py`)：
- ResNet-50 前三个 stage → 特征图 (B, 1024, 14, 14)
- 展平为 196 个 token → 线性投影到 256 维 → 加位置编码和 [CLS] token
- 4 层 Transformer 编码器（256 dim, 8 heads, GELU）
- [CLS] 输出 → AFF 融合数值特征 → FC → 预测值

**M3: 纯 ViT-Base 基线（次优先级，代码先生成，后续再跑）** (`src/models/vit_baseline.py`)：
- 用 timm 的 `vit_base_patch16_224`，ImageNet 预训练，修改输入通道为 9
- [CLS] token 输出 768 维 → AFF 融合数值特征 → FC → 预测值
- 目的：和 M2 对比，区分"提升来自 Transformer"还是"来自混合设计"
  - 如果 M3 < M1 < M2：证明混合设计本身是贡献（最佳结果）
  - 如果 M3 ≈ M2 > M1：说明功劳在 Transformer，CNN 部分可选
  - 如果 M3 > M2：说明纯 ViT 就够了，不需要混合

**M4: SatMAE 微调（最低优先级，代码先生成，后续探索用）** (`src/models/satmae_finetune.py`)：
- 尝试加载 SatMAE 预训练权重（https://github.com/sustainlab-group/SatMAE）
- 如果权重可用：ViT-Large 骨干 + SatMAE-Sentinel 权重，冻结前 20 层，微调最后 4 层 + 回归头
- 如果权重不可用：用 timm 的 `vit_large_patch16_224` + ImageNet 权重替代，留好 SatMAE 加载接口
- 回归头：AFF 融合数值特征 → FC → 预测值
- 目的：测试遥感预训练 vs ImageNet 预训练的差距，判断瓶颈在架构还是在预训练数据

**模型工厂** (`src/models/__init__.py`)：根据配置返回 M1/M2/M3/M4

### 模块五：训练

写 `src/train.py`：

- 损失：MSE
- 优化器：AdamW，差异化学习率
  - M1：预训练层 1e-4，AFF+头 5e-4
  - M2：CNN 层 1e-4，Transformer 层 1e-3，AFF+头 5e-4
  - M3：预训练层 1e-5，AFF+头 5e-4
  - M4：冻结层不更新，微调层 1e-5，AFF+头 5e-4
- 调度：CosineAnnealing + 5 epoch warmup
- 早停：验证 R² 连续 30 epoch 不提升
- 每 epoch 打印 train/valid 的 R² 和 loss
- 保存最佳权重
- 支持 --model 和 --mode 参数切换

### 模块六：评估和可视化

写 `src/evaluate.py`：
- 测试集 R²、RMSE、MAE
- 所有已训练模型的对比表格（先跑哪些就对比哪些）
- VIIRS 夜间灯光 OLS 回归作为基线
- 预测误差与初始条件的相关系数（确认无系统偏差）

写 `src/visualize.py`：
- 预测 vs 实际散点图（每个模型一张，含 level 和 diff 子图）
- 所有模型的 R² 对比柱状图
- M2 的 Transformer 注意力热力图（选几个测试样本可视化）

### 模块七：配置和项目结构

`configs/default.yaml` 集中管理所有超参数。

```
satellite-econ-guangdong/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   ├── 01_export_gee.py
│   └── 02_prepare_labels.py    # 含标签处理 + 数据划分
├── src/
│   ├── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── aff.py
│   │   ├── resnet_baseline.py     # M1 优先跑
│   │   ├── cnn_transformer.py     # M2 优先跑（主模型）
│   │   ├── vit_baseline.py        # M3 次优先
│   │   └── satmae_finetune.py     # M4 后续探索
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── data/
│   ├── raw/
│   │   └── guangdong_county_gdp_template.csv
│   ├── sentinel2/
│   └── processed/
└── outputs/
```

## 技术要求

- Python 3.10+, PyTorch 2.0+, timm, einops, rasterio, geemap, earthengine-api, matplotlib, scikit-learn, pandas, pyyaml, tqdm
- 超参数集中在 yaml，不硬编码
- 中文注释
- 代码结构清晰，M3 和 M4 的代码现在就生成好，后续可以直接跑，不需要改 train.py 或 dataset.py
- 如果后续要加新数据源（SAR、NO2）或新模型（Swin），只需加文件不需要改已有代码
