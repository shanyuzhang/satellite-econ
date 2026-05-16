# 卫星图像预测广东省经济活动 — 项目文档

> 探索 ViT+CNN 混合架构在 2.4km 网格级 GDP 预测上的可行性
> 数据：Sentinel-2 + VIIRS + 县级 GDP，区域：珠三角 9 市，时间：2018-2023

---

## 1. 项目背景与目的

### 1.1 这是个什么实验

参照两篇论文做的方法论组合：
- **Khachiyan et al. (2022, AER Insights)** —— 纯 CNN 预测美国 2.4km 网格收入和人口，R²₍水平值₎ 0.85-0.91，R²₍十年差分₎ 0.32-0.46
- **Cheng & Jiang (2024)** —— ResNet-50 + AFF（注意力特征融合）融合图像与能源数据，加州 58 县 R²₍水平值₎ 0.97

在它们基础上，我们尝试一个改进：**用 CNN→Transformer 串行混合架构替代纯 CNN**，验证 Transformer 的全局空间建模对经济预测是否有增量价值。

### 1.2 核心研究假设

> **假设**：CNN 抽局部纹理 + Transformer 建全局空间关系，应该比纯 CNN 或纯 ViT 都好。

具体来说，预期 4 个模型的相对性能为：
- 如果 M3 < M1 < M2 → 混合设计本身是贡献（最佳剧本）
- 如果 M3 ≈ M2 > M1 → 功劳在 Transformer，CNN 部分可选
- 如果 M3 > M2 → 纯 ViT 就够了，不需要混合

### 1.3 实验性质

**不是完整论文实验**——暂不追求消融、多 seed、跨地区。先把核心管线跑通、拿初步结果、看效果，再决定后续。

---

## 2. 使用的数据

### 2.1 卫星图像：Sentinel-2 L2A 表面反射率（Harmonized）

- **GEE collection**：`COPERNICUS/S2_SR_HARMONIZED`
- **9 个波段**：B2 (蓝)、B3 (绿)、B4 (红)、B5/B6/B7 (红边)、B8 (近红外)、B11/B12 (短波红外)
- **分辨率**：10m（导出时 scale=10）
- **时间合成**：每年 1-12 月全年中位数合成（5-8 月窗口在 PRD 雨季多云，部分边缘网格 0 影像 → 改全年）
- **云掩膜**：SCL 波段排除云阴影/云中/云高/卷云/雪（class 3,8,9,10,11）

### 2.2 辅助数据

- **GHSL SMOD**（`JRC/GHSL/P2023A/GHS_SMOD_V2-0`）：城镇化等级分类，用于过滤
  - 阈值 ≥21（郊区+）+ 网格内至少 10% 像素达标 → 排除纯山地/水域/农田
- **VIIRS DNB**（`NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`）：年度夜间灯光均值，作为数值特征 + OLS 基线
- **FAO GAUL 2015 level1/2**：广东省 + 21 个地级市行政边界

### 2.3 标签：县级 GDP

**数据来源**

来源为各市统计年鉴和统计公报（`data/raw/guangdong_county_gdp.csv`），手动整理。字段格式：

```
county_name,year,gdp_billion_yuan
```

`county_name` 使用 **FAO GAUL ADM2 的拼音拼写**（见 2.5 节匹配说明），以便与网格元数据直接 join。

**数据覆盖**：珠三角 9 市 × 2018-2023 年 = **54 行**，单位：十亿元人民币（billion yuan）。

**原始 GDP 数值**（单位：亿元 = 0.1 billion yuan，下表已转换为 billion yuan）：

| 市（county_name） | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---:|---:|---:|---:|---:|---:|
| Guangzhou | 2285.9 | 2362.9 | 2501.9 | 2823.0 | 2882.0 | 3049.0 |
| Shenzhen | 2422.2 | 2692.7 | 2767.0 | 3065.9 | 3236.0 | 3461.0 |
| Foshan | 993.6 | 1076.9 | 1078.8 | 1270.3 | 1270.9 | 1359.1 |
| Dongguan | 828.5 | 916.8 | 970.5 | 1108.1 | 1163.0 | 1167.9 |
| Zhongshan | 393.3 | 390.0 | 340.3 | 364.6 | 382.7 | 430.5 |
| Zhuhai | 291.5 | 339.0 | 330.7 | 373.3 | 398.0 | 432.0 |
| Huizhou | 401.8 | 429.1 | 450.3 | 478.1 | 502.0 | 527.9 |
| Zhaoqing | 226.3 | 248.0 | 247.8 | 283.0 | 296.0 | 315.0 |
| Jiangmen | 268.4 | 299.3 | 303.0 | 352.5 | 370.8 | 409.0 |

数值来自官方统计公报，精确到个位数（十亿元）。城市间 GDP 量级差异约 **10 倍**（深圳/广州 ~3000 vs 中山/珠海 ~300-400 亿元），这是后续讨论的核心问题之一。

**分摊方式**

城市 GDP 在该市所有网格上**均匀分配**：

```
GDP_grid = GDP_city / N_grids_in_city
```

这是已知的简化假设（每个网格经济活动完全相同），Khachiyan (2022) 使用的是人口权重插值，我们暂不做。`N_grids_in_city` 由 `grid_meta.csv` 动态计算，即 GHSL 过滤后每市实际保留的网格数。

### 2.4 数据规模

| 项 | 数量 |
|---|---|
| 珠三角 9 市总网格（2.4km） | ~4885 |
| 剔除 UNKNOWN 县（海岸边界外）后 | 4730 |
| × 6 年 | **28,380 个 GeoTIFF** |
| 实际可用样本（level 模式） | ~28,380（每个 grid×year 一样本） |
| 实际可用样本（diff1 模式） | ~23,650（第一年无前一年差分，去掉 2018） |
| 总数据量 | ~55 GB |

---

## 3. 图像生成详解：每张 GeoTIFF 是怎么来的

### 3.1 网格定义

**坐标系**：WGS84 (EPSG:4326)，经纬度坐标。

**网格尺寸**：2400m × 2400m。在 GEE 中用经纬度量化实现：

```python
# 以 2400m ≈ 0.02165° 经度 / 0.02165° 纬度 在珠三角纬度下
grid_width_deg  = grid_size_m / 111320        # ≈ 0.02157°
grid_height_deg = grid_size_m / 110540        # ≈ 0.02172°

# 把每个像素的经纬度向下取整到格子边界
lon_quantized = floor(lon / grid_width_deg)  * grid_width_deg
lat_quantized = floor(lat / grid_height_deg) * grid_height_deg
grid_id = format("{lon_q:.4f}_{lat_q:.4f}")
```

这是**服务器端**操作（`ee.Image.pixelLonLat()` + 数学运算），避免了把 30k 个 Python `ee.Feature` 对象序列化传给 GEE API（会触发 10MB 负载限制）。

**覆盖范围**：FAO GAUL 2015 广东省（`ADM1_NAME == "Guangdong Sheng"`）的 bounding box，再取 region_subset（珠三角 9 市）的 union 几何与网格交集。

### 3.2 城镇化过滤（GHSL SMOD）

不是所有 2.4km 网格都被保留。过滤条件：

```
网格内像素的 GHSL SMOD 等级 ≥ 21（郊区 Suburban）的占比 ≥ 10%
```

SMOD 等级对照：

| 等级 | 含义 |
|---:|---|
| 10 | 水体 |
| 11/12 | 农村低密度 |
| 13 | 乡村集群 |
| **21** | **郊区（阈值）** |
| 22 | 半密集城市 |
| 23 | 密集城市 |
| 30 | 城市中心 |

这个过滤的目的：去掉纯山地、农田、水库等经济活动极少的网格，避免"用卫星图学到山的纹理来预测 GDP"的干扰信号。过滤后从约 4885 个候选网格保留约 4730 个（另有 155 个位于 GAUL ADM2 边界之外或海上，归为 UNKNOWN 后剔除）。

### 3.3 县级标注（assign county）

每个网格中心点落入哪个 GAUL ADM2 多边形，就属于哪个市。GEE 服务器端实现：

```python
grid_points = grid_fc.map(lambda f: f.centroid())
joined = ee.Join.saveFirst("county_props").apply(
    primary=grid_points,
    secondary=counties_fc,        # FAO GAUL ADM2，过滤到 9 市
    condition=ee.Filter.intersects(".geo", ".geo")
)
```

输出 `county_name`（ADM2_NAME，纯拼音如 "Guangzhou"、"Shenzhen"）和 `county_code`（GAUL ADM2_CODE）。注意：**GAUL ADM2_CODE 与国家统计局六位行政代码不对应**，因此后续所有 join 全部用 `county_name` 拼音完成。

### 3.4 Sentinel-2 合成：每个 grid×year 一张 GeoTIFF

对每个 `(grid_id, year)` 组合，GEE 流程为：

```
S2_SR_HARMONIZED 集合
  → 过滤时间窗口：{year}-01-01 ~ {year}-12-31  （全年，非 5-8 月）
  → 过滤 bounding box：该 2.4km 网格
  → SCL 云掩膜：排除 class 3（云阴影）、8（云中）、9（云高）、10（卷云）、11（雪）
  → 选 9 波段：B2 B3 B4 B5 B6 B7 B8 B11 B12
  → 时间维 median()（逐像素取中位数，鲁棒去云）
  → reproject(EPSG:4326, scale=10)
  → Export.image.toDrive(
         folder="gee_guangdong_s2",
         fileNamePrefix="{grid_id}_{year}",
         region=grid_geometry,
         scale=10,
         crs="EPSG:4326",
         fileFormat="GeoTIFF"
     )
```

**为什么全年合成，不用 5-8 月？**
珠三角雨季集中在 4-9 月，2018/2019 年部分边缘网格 5-8 月窗口内有效影像数为 0（全部被 SCL 掩掉），导致 GeoTIFF 全为 NoData。改为全年合成后，即使雨季云多，1-4 月和 10-12 月的晴天影像仍能提供有效像素。

**导出参数**：
- `scale=10`（Sentinel-2 原生分辨率），每张图约 240×240 像素（10m × 240 = 2400m）
- 实际读取时 `resize` 到 224×224（dataset.py 中双线性插值）
- 存储：每张约 2 MB（9 波段 × float32），约 28,380 张 × 2 MB ≈ **55 GB**

**文件命名**：`{grid_id}_{year}.tif`，例如 `113.6522_22.8043_2021.tif`。

### 3.5 VIIRS 年度均值

同步从 `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` 导出，对同一网格取全年 12 个月均值，写入 `viirs_*.csv`（按市或批次分批导出，之后在 Colab 里合并成 `viirs.csv`）。

输出格式：
```
grid_id, year, viirs_mean   （单位：nW/cm²/sr）
```

VIIRS 作为：① 数值特征（送入 AFF 模块）；② OLS 基线的输入特征。

---

## 4. 经济指标与图像的匹配

### 4.1 匹配的核心挑战

GDP 是**市级年度数据**（1 个城市 1 个数），但我们的图像是**网格级年度数据**（4730 个网格）。匹配需要解决两个问题：

1. **空间粒度不匹配**：1 个市 → N 个网格（均匀分摊）
2. **编码不一致**：NBS 行政代码（6 位）≠ GAUL ADM2_CODE → 只能用拼音名字 join

### 4.2 匹配流程（详细步骤）

```
输入文件：
  guangdong_county_gdp.csv  → county_name（拼音）, year, gdp_billion_yuan
  grid_meta.csv             → grid_id, county_name（GAUL 拼音）, county_code, lat, lng
  viirs.csv                 → grid_id, year, viirs_mean
```

**Step 1：统一 county_name 格式**

```python
gdp_df["_cn"]    = gdp_df["county_name"].str.strip().str.lower()
grid_meta["_cn"] = grid_meta["county_name"].str.strip().str.lower()
# "Guangzhou" → "guangzhou"，消除大小写和空白差异
```

GDP 表的 county_name 必须与 GAUL ADM2_NAME 完全对应（如 "Guangzhou"、"Shenzhen"、"Foshan" 等），不要用中文或 NBS 代码。

**Step 2：统计每市的网格数**

```python
grids_per_county = grid_meta.groupby("_cn").size().rename("n_grids")
# 例如：guangzhou → 1250 grids，shenzhen → 810 grids，...
```

**Step 3：均匀分摊**

```python
merged = gdp_df.merge(grids_per_county, on="_cn")
merged["gdp_per_grid"] = merged["gdp_billion_yuan"] / merged["n_grids"]
```

**Step 4：关联到每个网格**

```python
out = grid_meta[["grid_id", "_cn"]].merge(
    merged[["_cn", "year", "gdp_per_grid"]], on="_cn"
)
```

结果：`(grid_id, year) → gdp_billion_yuan_grid`，共 4730 × 6 = 28,380 行。

**Step 5：计算对数 GDP 和差分**

```python
grid_gdp["log_gdp"]      = np.log(grid_gdp["gdp_billion_yuan_grid"])
grid_gdp["diff_log_gdp"] = grid_gdp.groupby("grid_id")["log_gdp"].diff()
# diff_log_gdp ≈ 年度 GDP 增长率（对数差 ≈ 百分比增长）
# 2018 年的 diff_log_gdp 为 NaN（没有 2017 年数据），diff1 模式下会自动丢弃
```

**Step 6：基期控制变量**

```python
base = gdp_df[gdp_df["year"] == 2018][["county_name", "log_gdp_county"]]
# base_log_gdp = 该市 2018 年的 log(GDP)（市级，非网格级）
# 用途：控制城市间初始发展水平差异
```

注意：`base_log_gdp` 是**市级总 GDP 的对数**，而非分摊后的网格级 GDP。同一市内所有网格的 `base_log_gdp` 完全相同。在 Iteration 1 实验中该特征实际上是城市身份的代理变量，可能导致过拟合。

**Step 7：VIIRS 合并**

```python
df = grid_gdp.merge(viirs, on=["grid_id", "year"], how="left")
# 以 left join 保留所有网格，VIIRS 缺测（边缘网格偶发）的置 NaN
# 训练时 DataLoader 中 NaN 样本会被跳过（dropna 或 nanmean 处理）
```

**Step 8：z-score 标准化（仅用训练集统计量）**

```python
for col in ["base_log_gdp", "viirs_mean"]:
    mean = df.loc[train_mask, col].mean()
    std  = df.loc[train_mask, col].std()
    df[f"{col}_z"] = (df[col] - mean) / std
# 统计量保存到 feature_stats.json，推断时复用
```

### 4.3 最终 labels.csv 的结构

```
grid_id          → 网格标识（经纬度编码字符串）
county_name      → 所属市（GAUL 拼音）
year             → 年份（2018-2023）
gdp_billion_yuan_grid  → 分摊后网格 GDP（原始值）
log_gdp          → ln(gdp_billion_yuan_grid)，level 模式的预测目标
diff_log_gdp     → log_gdp[t] - log_gdp[t-1]，diff1 模式的预测目标（2018 为 NaN）
base_log_gdp     → 2018 年市级 log GDP（控制变量，同市内相同）
viirs_mean       → 年度 VIIRS 均值（nW/cm²/sr）
base_log_gdp_z   → base_log_gdp 的 z-score（用训练集统计量）
viirs_z          → viirs_mean 的 z-score（用训练集统计量）
split            → "train" / "valid" / "test"
```

---

## 5. 训练/验证/测试划分详解

### 5.1 划分的核心约束

**必须保证**：同一个 `grid_id` 的所有年份（2018-2023）必须全部分到同一个 split，否则模型在训练集里"见过"某个网格的图像，在测试集里预测同一网格的另一年，会造成严重的时间泄漏。

**额外约束**：是否要求同一个市的所有网格都在同一 split（**按市划分 vs 按网格划分**），这是两种策略的核心区别，也是 Iteration 1 失败、Iteration 2 修复的关键。

### 5.2 Iteration 1：按市（county）划分——结果全负

**方法**：把 9 个市随机分为 train 5 / valid 2 / test 2。比例大致为 60/20/20，但实际单位是"市"（9 个）而非网格（4730 个）。

```python
def split_by_county(county_names, train_r, valid_r, test_r, seed):
    rng = np.random.RandomState(seed)
    counties = sorted(set(county_names))
    rng.shuffle(counties)           # 共 9 个市
    n_train = int(9 * 0.6) = 5
    n_valid = int(9 * 0.2) = 1      # 实际 1 个市
    n_test  = 9 - 5 - 1 = 3        # 实际 3 个市
    # 结果：某次划分可能是：
    # train: Guangzhou, Shenzhen, Foshan, Dongguan, Zhuhai
    # valid: Huizhou
    # test:  Zhongshan, Zhaoqing, Jiangmen
```

**为什么失败？**

按市划分是"严格防泄漏"策略，正确性上无懈可击——但在只有 9 个城市的数据集上是灾难性的：

- 城市间 GDP 量级差异约 10 倍（深圳/广州 ≈ 3000 亿 vs 肇庆 ≈ 300 亿）
- 模型在训练集里只见过 5 个城市的量级
- 测试集的城市（如肇庆）GDP 量级可能完全超出训练集分布
- 预测时模型不知道"肇庆 GDP 约为深圳的 10%"这一信息，胡乱预测

**结果**：level 模式 R² ≈ −27（比预测均值还差 27 倍），diff1 模式虽好一些但 R² 仍为负。

```
Iteration 1 结果（by_county, level mode）：
  M1 ResNet-50   R² = −27.79  RMSE = 1.79  MAE = 1.75
  M2 混合        R² = −26.64  RMSE = 1.75  MAE = 1.71
  OLS (VIIRS)    R² = −23.06  RMSE = 1.49  MAE = 1.45
```

### 5.3 Iteration 2：按网格（grid）随机划分——当前方案

**方法**：把 4730 个 `grid_id` 随机打乱，按 60/20/20 分配到 train/valid/test。同一 `grid_id` 的所有年份（6 个样本）始终在同一 split，保证时间不泄漏。同一市内不同 `grid_id` 可能跨 split。

```python
def split_by_grid(grid_ids, train_r=0.6, valid_r=0.2, test_r=0.2, seed=42):
    rng = np.random.RandomState(seed)
    gids = sorted(set(grid_ids))   # 4730 个唯一 grid_id
    rng.shuffle(gids)
    n = len(gids)
    n_train = int(n * 0.6)   # 2838 个网格 → 17028 个样本
    n_valid = int(n * 0.2)   # 946 个网格  → 5676 个样本
    n_test  = n - n_train - n_valid  # 946 个网格 → 5676 个样本
    mapping = {g: split for i, g in enumerate(gids)
               for split in [["train","valid","test"][
                   0 if i < n_train else 1 if i < n_train+n_valid else 2]]}
    return mapping
```

**实际划分数量**（2018-2023 level 模式）：

| split | 网格数 | 样本数（×6年） |
|---|---:|---:|
| train | 2838 | ~17028 |
| valid | 946  | ~5676  |
| test  | 946  | ~5676  |

**与 Iteration 1 的关键区别**：

| 维度 | by_county | by_grid（当前） |
|---|---|---|
| 划分粒度 | 市级（9 个单位） | 网格级（4730 个单位） |
| 同市网格是否跨 split | 不会 | **会**（同市约 60/20/20） |
| 量级泄漏风险 | 无（最严格） | 有（训练集见过同市其他网格） |
| level 模式可学习性 | 极低（城市未见过） | 高（同城市量级在训练集出现） |
| 泛化场景对应 | 完全新城市的预测 | 同城市内未见过网格的预测 |

**当前方案的权衡**：

按网格划分牺牲了"完全新城市"的外样本泛化能力，但在只有 9 个城市的数据集规模下，这是实验可行性的必要妥协。一旦扩展到全广东 21 个市（>100 个行政区，500+ 个 county），可以恢复 by_county 划分并期望得到合理结果。

**配置方法**（`configs/default.yaml`）：

```yaml
data:
  split:
    mode: random          # by_county | random
    train_ratio: 0.6
    valid_ratio: 0.2
    test_ratio: 0.2
    seed: 42
```

运行 `python scripts/02_prepare_labels.py` 会输出 `data/processed/split.json`，格式：

```json
{
  "train": ["113.6522_22.8043", "113.6522_22.8260", ...],  // 2838 个 grid_id
  "valid": [...],   // 946 个
  "test":  [...]    // 946 个
}
```

---

## 6. 整体管线架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 1: 数据获取 (GEE, 一次性, 1-3 天异步)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  scripts/01_export_gee.py                                                │
│    ├ 加载 FAO GAUL 行政边界                                              │
│    ├ 服务器端构造 2400m 网格 (ee.List.sequence + map)                    │
│    ├ GHSL SMOD 过滤城镇化                                                │
│    ├ 给每个网格分配 ADM2 county 信息                                     │
│    ├ 提交 ~28k 个 Export.image.toDrive 任务（带配额节流 + 重试）         │
│    └ 同时导出 VIIRS 年度均值 + 网格元数据                                │
│  → 输出：Google Drive 的 gee_guangdong_s2/ 文件夹                        │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 2: 标签准备 (Colab, 2 分钟)                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  scripts/02_prepare_labels.py                                            │
│    ├ 合并 viirs_*.csv → viirs.csv                                        │
│    ├ 县 GDP / 网格数 → 网格级 GDP (均匀分摊)                             │
│    ├ 计算 log_gdp, diff_log_gdp                                          │
│    ├ 基期 (2018) 县级 log_gdp 作为初始条件控制变量                       │
│    ├ 按 grid_id 随机划分 train/valid/test = 60/20/20（同 grid 所有年同 split）│
│    └ 训练集统计量做 z-score 标准化                                       │
│  → 输出：labels.csv, split.json, feature_stats.json                      │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 3: 训练 (Colab A100, 每个 run 1-2 小时)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  src/dataset.py: GuangdongGridDataset                                    │
│    ├ rasterio 读 GeoTIFF (9, H, W)                                       │
│    ├ resize 到 224×224                                                   │
│    ├ 训练集逐波段 z-score（一次性扫 500 样本估统计量）                   │
│    ├ 随机翻转 + 90° 旋转（仅 train）                                     │
│    └ 输出 {image: (9,224,224), features: (num_feat,), label, grid_id}    │
│                                                                          │
│  src/models/                                                             │
│    ├ aff.py: AFF 注意力特征融合 (借鉴 Cheng & Jiang)                     │
│    ├ resnet_baseline.py: M1 = 9 通道 ResNet-50 + AFF                     │
│    ├ cnn_transformer.py: M2 = ResNet 前 3 stage → Transformer + AFF      │
│    ├ vit_baseline.py: M3 = ViT-Base (timm) + AFF                         │
│    └ satmae_finetune.py: M4 = ViT-Large + SatMAE 微调 + AFF (待跑)       │
│                                                                          │
│  src/train.py                                                            │
│    ├ MSE 损失                                                            │
│    ├ AdamW + 差异化学习率（backbone vs head）                            │
│    ├ Linear warmup + Cosine annealing                                    │
│    ├ 早停: valid R² 连续 30 epoch 不提升                                 │
│    ├ wandb 实时记录                                                      │
│    └ 保存最佳 checkpoint                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 4: 评估 (Colab T4, 5 分钟)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  src/evaluate.py                                                         │
│    ├ 测试集 R²/RMSE/MAE                                                  │
│    ├ VIIRS+base_log_gdp OLS 基线对比                                     │
│    └ 残差与初始条件的相关性 (诊断系统偏差)                               │
│                                                                          │
│  src/visualize.py                                                        │
│    ├ 预测 vs 实际散点图                                                  │
│    ├ R² 柱状图                                                           │
│    └ M2 Transformer CLS→patch 注意力热力图 (叠加 RGB)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 模型架构详解

四个模型**共同的接口**：
```
image: (B, 9, 224, 224)        ← 9 波段 Sentinel-2
features: (B, num_features)    ← VIIRS + base_log_gdp（数值特征）
→ pred: (B, 1)                 ← 预测的 log GDP（或 diff_log_gdp）
```

四个模型**共用的组件**：**AFF**（Attentional Feature Fusion）—— 把图像 backbone 输出的高维特征向量和数值特征向量加权融合（而非简单拼接）。借鉴 Cheng & Jiang (2024) 在能源-图像融合的做法。

### 7.1 M1: ResNet-50 基线（纯 CNN）

```
图像 (9, 224, 224)
  ├ Conv1 (9→64): 前 3 通道复用 RGB 预训练权重，后 6 通道用 RGB 均值初始化
  ├ ResNet-50 layer1 ~ layer4 (ImageNet 预训练)
  ├ Global Average Pool → 2048 维
  └ AFF 融合数值特征 → FC(256) → FC(1)
```

### 7.2 M2: CNN→Transformer 混合 (主模型)

```
图像 (9, 224, 224)
  ├ Conv1 (9→64) + ResNet-50 layer1~layer3 (ImageNet 预训练)
  │    输出 (1024, 14, 14) 特征图
  ├ Flatten + Linear(1024→256) → 196 个 token
  ├ + [CLS] token + 可学习位置编码 → 197 tokens
  ├ Transformer Encoder × 4 (d=256, 8 heads, GELU, norm_first=True)
  ├ 取 CLS 输出 (256 维)
  └ AFF 融合数值特征 → FC(256) → FC(1)
```

**核心动机**：
- CNN 前 3 stage 保留 14×14 空间结构（不像 ResNet 全做 GAP），可以让 Transformer 在 patch 级别学全局关系
- 14×14=196 个 token 接近 ViT-Base 的 196 patch 数，但每个 token 的语义比 ViT 的"原始 16×16 像素块"更丰富（已经过 CNN 抽特征）

### 7.3 M3: 纯 ViT-Base 基线

```
图像 (9, 224, 224)
  ├ timm.vit_base_patch16_224 (ImageNet 预训练, in_chans=9)
  │   patch 16×16 → 196 patch + CLS → 12 层 Transformer
  ├ 取 CLS 输出 (768 维)
  └ AFF 融合数值特征 → FC(256) → FC(1)
```

**目的**：跟 M2 对比，分离"提升来自 Transformer"还是"来自混合架构"。

### 7.4 M4: SatMAE 微调（次优先级，已生成代码未跑）

```
图像 (9, 224, 224)
  ├ ViT-Large (timm, in_chans=9)
  │   优先加载 SatMAE 在 Sentinel-2 上的预训练权重（手动放到 weights/satmae_vit_large.pth）
  │   找不到则回退到 ImageNet 预训练
  ├ 冻结前 20 个 Transformer block，只微调后 4 个 + 回归头
  └ AFF 融合 → FC → FC(1)
```

**目的**：测试遥感专有预训练 vs ImageNet 预训练的差距。

### 7.5 训练目标的两种模式

- **level 模式**：直接预测 `log_gdp`（绝对水平）
- **diff1 模式**：预测 `diff_log_gdp = log_gdp[t] - log_gdp[t-1]`（年度增长率）

Khachiyan 论文的差分版本是十年差分；我们这里数据只有 6 年，用 1 年差分（diff1）。

---

## 8. 训练细节

| 配置 | 值 |
|------|---|
| 损失 | MSE |
| 优化器 | AdamW (weight_decay=0.05) |
| 调度器 | LinearLR warmup (5 epoch) + CosineAnnealingLR |
| Batch size | 128 (A100) / 64 (V100) |
| Epoch 上限 | 200 |
| 早停 | valid R² 连 30 epoch 不提升 |
| 梯度裁剪 | grad_norm ≤ 1.0 |
| 划分 | 60/20/20，按 grid_id 随机（同 grid 所有年同 split） |
| 数据增强 | 随机水平/垂直翻转 + 90° 旋转 (仅 train) |
| 随机种子 | 42 |

### 差异化学习率

不同模块用不同 lr，防止预训练权重被高 lr 冲掉：

```yaml
M1: {backbone: 1e-4, head: 5e-4}                          # ResNet 预训练
M2: {cnn: 1e-4, transformer: 1e-3, head: 5e-4}            # CNN 慢，Transformer 从头训
M3: {backbone: 1e-5, head: 5e-4}                          # ViT 整体预训练，更小 lr
M4: {finetune: 1e-5, head: 5e-4, freeze_layers: 20}       # 冻一半，剩下微调
```

---

## 9. 当前结果（2018-2023, 珠三角 9 市）

### 9.1 level 模式（log_gdp 绝对水平）

| 模型 | R² | RMSE | MAE | n |
|------|------:|------:|------:|---:|
| M1 ResNet-50 | −27.79 | 1.79 | 1.75 | 14952 |
| M2 混合 | −26.64 | 1.75 | 1.71 | 14952 |
| M3 ViT-Base | (类似) | | | |
| OLS (VIIRS+base) | −23.06 | 1.49 | 1.45 | 8544 |

**level 模式全军覆没**。所有模型 R² < −20，连 OLS 基线都是负数。**核心问题**：城市间 GDP 量级跨度太大（深圳/广州 vs 肇庆/江门差 10 倍），按市划分 train/valid/test 之后，验证集的城市是模型完全没见过量级的，预测必然跑飞。

### 9.2 diff1 模式（年度差分）

| 模型 | R² | RMSE | MAE | n |
|------|------:|------:|------:|---:|
| OLS (VIIRS+base) | −0.146 | 0.025 | 0.018 | 4272 |
| M1 ResNet-50 | −0.068 | 0.047 | 0.040 | 10680 |
| M2 混合 | +0.009 | 0.045 | 0.038 | 10680 |
| **M3 ViT-Base** | **+0.191** | 0.041 | 0.031 | 10680 |

**diff1 模式有正 R²**！差分消除了城市间基础量级差异，模型学到的是"经济增长信号"。

**结果出乎意料**：纯 ViT (M3, R²=0.19) > 混合 (M2, R²=0.01) > 纯 CNN (M1, R²=−0.07)

对照三种剧本：
- ❌ M3 < M1 < M2 不成立
- ❌ M3 ≈ M2 > M1 不成立
- ✅ **M3 > M2 ≈ M1** —— **纯 ViT 就够了，混合的 CNN 部分反而是累赘**

### 9.3 残差与初始条件的相关性（诊断）

```
M1: corr(residual, base_log_gdp) = −0.16
M2: corr(residual, base_log_gdp) = −0.19
M3: corr(residual, base_log_gdp) = −0.14
```

负相关说明所有模型都**在高 GDP 区域 under-predict、低 GDP 区域 over-predict**，存在轻微的"回归到均值"偏差。但绝对值不大（0.14-0.19），不是主要问题。

---

## 10. 结果的意义和局限

### 10.1 这次实验告诉我们什么

✅ **管线工作**：从 GEE 导出 → 标签 → 训练 → 评估 → 可视化全部跑通，A100 上单 run ~1 小时。

✅ **diff1 模式可行**：在差分预测上，模型超过了 OLS 基线和"预测均值"基线，证明卫星图像确实携带经济增长信号。

✅ **初步否定假设**：M3 > M2 > M1 的结果表明 Transformer 是关键，CNN 局部特征反而干扰。这是个**有意义的负面结果**——值得进一步验证。

### 10.2 已知局限

⚠️ **样本量小**：只有 9 个市，by-city 划分等于 train 5 / valid 2 / test 2，量级跨域问题严重。
⚠️ **GDP 分摊太粗**：均匀分配假设每个网格经济活动相同，实际差异巨大。需要人口权重插值。
⚠️ **训练时长**：30-60 epoch 早停，可能没收敛到最优。
⚠️ **单 seed**：所有结论基于 seed=42，没有方差估计。

### 10.3 建议的下一步

**短期（1-2 天可做完）**：

1. **改成 grid-level random split**：放弃严格 by-city 划分，按 grid 随机分组（牺牲一些纯净性换可学习的训练集）→ 预期 R² 大幅好转到 0.3-0.6
2. **改预测目标为 GDP density**（亿元/km²）：减少城市间量级跨度，level 模式可能恢复
3. **跑 M4 SatMAE**：验证遥感预训练 vs ImageNet 预训练
4. **多 seed 重复**（seed=42/0/1）：估计方差

**中期（1 周可做完）**：

5. **扩到全广东 21 个市**：by-city 划分时训练集有 12 个市，量级覆盖更全
6. **加 SAR (Sentinel-1) 数据**：补充全天候信息
7. **加更多控制变量**：人口密度、建成区面积

**长期**：

8. **人口权重插值**（Khachiyan 范式）替代均匀分摊
9. **跨地区泛化测试**：广东训练 → 浙江测试
10. **跨年泛化**：2018-2021 训练 → 2022-2023 测试

---

## 11. 代码组织

```
satellite-econ-guangdong/
├── README.md
├── docs/
│   └── PROJECT.md                  ← 本文档
├── configs/
│   └── default.yaml                ← 集中超参数（路径、波段、训练参数、模型选择）
├── scripts/
│   ├── 01_export_gee.py            ← GEE Sentinel-2 + VIIRS 异步导出
│   └── 02_prepare_labels.py        ← 标签处理 + 数值特征 + 划分
├── src/
│   ├── dataset.py                  ← PyTorch Dataset
│   ├── train.py                    ← 训练入口（差异化 lr + warmup+cosine + 早停）
│   ├── evaluate.py                 ← 测试集评估 + OLS 基线
│   ├── visualize.py                ← 散点图 / 柱状图 / 注意力热力图
│   ├── utils.py                    ← seed / metrics / io
│   └── models/
│       ├── aff.py                  ← AFF 注意力特征融合
│       ├── resnet_baseline.py      ← M1
│       ├── cnn_transformer.py      ← M2（主模型）
│       ├── vit_baseline.py         ← M3
│       └── satmae_finetune.py      ← M4
├── notebooks/
│   └── colab_run.ipynb             ← Colab 端到端 notebook
├── data/
│   ├── raw/guangdong_county_gdp.csv
│   └── processed/                  ← labels/split/stats（自动生成）
└── outputs/
    ├── checkpoints/                ← 训练好的模型权重
    ├── logs/                       ← train/valid 指标 CSV
    └── figures/                    ← 散点图、柱状图、注意力热力图
```

---

## 12. 一句话总结

**目的**：验证 CNN→Transformer 混合架构在卫星-经济预测上的可行性。
**做了什么**：珠三角 9 市 × 6 年的 Sentinel-2 数据，跑通 ResNet / 混合 / ViT 三个模型在 level 和 diff1 两种目标上。
**得到了什么**：在差分模式 (diff1) 上，**纯 ViT 最好 (R²=0.19)、混合架构持平基线、纯 CNN 略差**——初步否定了"混合更好"的原假设，证据指向"Transformer 是关键，CNN 部分可去"。
**下一步**：放宽划分策略 (grid-level random)、改预测目标 (GDP density)、跑 M4 SatMAE 验证遥感预训练价值。
