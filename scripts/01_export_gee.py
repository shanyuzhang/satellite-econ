"""
01_export_gee.py — 用 Google Earth Engine 导出广东省 Sentinel-2 年度合成图像 + VIIRS 灯光

流程：
1. 加载广东省行政边界（FAO GAUL ADM1），生成 2.4km 网格
2. 用 GHSL SMOD 过滤掉纯山地/水域/森林网格，保留城镇化网格
3. 对每个 (grid, year)：S2 5-8 月、SCL 云掩膜、中位数合成、9 波段、scale=10
4. 异步提交 Export.image.toDrive；脚本监控任务状态
5. VIIRS 年度均值 / 网格元数据 用 reduceRegions 同步导出 CSV

用法：
    python scripts/01_export_gee.py --config configs/default.yaml --years 2015-2023
    python scripts/01_export_gee.py --years 2015 --max-tasks 5 --dry-run
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

try:
    import ee
except ImportError:
    print("[错误] 未安装 earthengine-api。请 pip install earthengine-api")
    sys.exit(1)


# -----------------------
# 重试装饰器：抗瞬时网络/SSL 断开（电脑休眠唤醒、WiFi 切换等）
# -----------------------
def with_retry(fn, max_retries: int = 8, base_delay: float = 2.0,
               max_delay: float = 60.0, what: str = ""):
    """
    用指数退避 + 抖动重试。遇到 ConnectionError/SSLError/HttpError/EEException 都重试。
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_name = type(e).__name__
            # 致命错误（非瞬时），直接抛
            fatal_msgs = ("not found", "permission denied", "no project found",
                          "Asset not found", "EOFError")
            msg = str(e)
            if any(m.lower() in msg.lower() for m in fatal_msgs):
                raise
            # 最后一次也失败 → 抛
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1.0)
            print(f"     [重试 {attempt+1}/{max_retries}] {what}: {err_name}: "
                  f"{msg[:120]}... 等 {delay:.1f}s 再试")
            time.sleep(delay)


# -----------------------
# 工具
# -----------------------
def parse_years(s: str):
    """支持 '2015-2023' 或 '2015,2017,2019' 或 '2020'"""
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_ee(project: Optional[str] = None):
    """初始化 GEE。
    GEE 现在强制要求 Cloud Project ID（2023+）。优先级：
      1. 函数参数 project
      2. 环境变量 EARTHENGINE_PROJECT
      3. config 里的 data.gee.project（由调用方传入）
    需要先 `earthengine authenticate` 一次。
    """
    project = project or os.environ.get("EARTHENGINE_PROJECT")
    if not project:
        raise RuntimeError(
            "未提供 GEE Cloud Project ID。\n"
            "请先：(1) 在 https://console.cloud.google.com/projectcreate 创建项目；\n"
            "      (2) 在 https://code.earthengine.google.com/register 把项目注册到 GEE；\n"
            "      (3) 设置环境变量 EARTHENGINE_PROJECT=<your-project-id> "
            "或在 configs/default.yaml 的 data.gee.project 填入。"
        )
    try:
        ee.Initialize(project=project)
    except Exception:
        print(f"[提示] GEE 初始化失败，尝试 ee.Authenticate() 后重试（project={project}）...")
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"[GEE] 已初始化，project={project}")


# -----------------------
# 网格生成
# -----------------------
def get_guangdong_geometry(admin1_name: str, region_subset: Optional[list] = None):
    """
    从 FAO GAUL 2015 取目标区域。
      - 若 region_subset 为 None：返回整个 ADM1（广东省）的几何
      - 若 region_subset 是 list：返回 GAUL ADM2 里这些地级市的并集几何
    """
    gaul1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    china = gaul1.filter(ee.Filter.eq("ADM0_NAME", "China"))

    province = china.filter(ee.Filter.eq("ADM1_NAME", admin1_name))
    n = province.size().getInfo()
    if n == 0:
        all_names = sorted(china.aggregate_array("ADM1_NAME").getInfo())
        raise RuntimeError(
            f"在 FAO/GAUL/2015/level1 中找不到 '{admin1_name}'。\n"
            f"中国所有 ADM1_NAME：\n  " + "\n  ".join(all_names)
        )
    print(f"     匹配 ADM1 '{admin1_name}'：{n} 个 feature")

    if not region_subset:
        return province.geometry()

    # 子集模式：基于 GAUL ADM2
    print(f"     启用 region_subset：{region_subset}")
    gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2") \
        .filter(ee.Filter.eq("ADM1_NAME", admin1_name)) \
        .filter(ee.Filter.inList("ADM2_NAME", region_subset))
    n2 = gaul2.size().getInfo()
    if n2 != len(region_subset):
        # 列出实际匹配的，便于排错
        matched = gaul2.aggregate_array("ADM2_NAME").getInfo()
        all2 = sorted(ee.FeatureCollection("FAO/GAUL/2015/level2")
                      .filter(ee.Filter.eq("ADM1_NAME", admin1_name))
                      .aggregate_array("ADM2_NAME").getInfo())
        missing = sorted(set(region_subset) - set(matched))
        raise RuntimeError(
            f"region_subset 中 {len(missing)} 个名字未在 GAUL ADM2 找到：{missing}\n"
            f"该 ADM1 下所有 ADM2_NAME：\n  " + "\n  ".join(all2)
        )
    print(f"     匹配 ADM2 子集：{n2} 个地级市")
    return gaul2.geometry()


def build_grid(province_geom, grid_size_m: int):
    """
    在广东省范围内生成 grid_size_m × grid_size_m 网格 —— **完全服务器端构造**。

    早期 Python 端用 list 累积 ee.Feature 的写法会把几万个 feature 全部序列化进
    单次请求，超过 GEE 10MB payload 上限。改成 ee.List.sequence + map，
    Python 端只描述"怎么算"，实际算在 GEE 服务器上。

    返回 FeatureCollection，每个 feature 带 grid_id / center_lng / center_lat / cell_idx。
    """
    step_deg = grid_size_m / 111000.0  # 1° ≈ 111 km 近似

    bounds = ee.List(province_geom.bounds().coordinates().get(0))
    sw = ee.List(bounds.get(0))   # 左下角
    ne = ee.List(bounds.get(2))   # 右上角
    minx = ee.Number(sw.get(0))
    miny = ee.Number(sw.get(1))
    maxx = ee.Number(ne.get(0))
    maxy = ee.Number(ne.get(1))

    nx = maxx.subtract(minx).divide(step_deg).ceil()
    ny = maxy.subtract(miny).divide(step_deg).ceil()

    def make_row(j):
        j = ee.Number(j)
        y0 = miny.add(j.multiply(step_deg))
        y1 = y0.add(step_deg)

        def make_cell(i):
            i = ee.Number(i)
            x0 = minx.add(i.multiply(step_deg))
            x1 = x0.add(step_deg)
            cell_idx = j.multiply(nx).add(i)
            return ee.Feature(
                ee.Geometry.Rectangle([x0, y0, x1, y1]),
                {
                    "cell_idx": cell_idx,
                    "center_lng": x0.add(step_deg / 2),
                    "center_lat": y0.add(step_deg / 2),
                },
            )

        return ee.List.sequence(0, nx.subtract(1)).map(make_cell)

    nested = ee.List.sequence(0, ny.subtract(1)).map(make_row)
    flat = nested.flatten()
    fc = ee.FeatureCollection(flat).filterBounds(province_geom)
    # 稳定可读的 grid_id（基于 cell_idx，与初始网格一一对应）
    fc = fc.map(lambda f: f.set(
        "grid_id",
        ee.String("g").cat(ee.Number(f.get("cell_idx")).format("%07d")),
    ))
    return fc


def assign_county(grid_fc, admin1_name: str):
    """
    用 FAO GAUL ADM2（县级）给每个 grid 打上 county_code 和 county_name。
    取 grid 中心点所在的 ADM2 polygon。
    """
    adm2 = ee.FeatureCollection("FAO/GAUL/2015/level2") \
        .filter(ee.Filter.eq("ADM1_NAME", admin1_name))

    def add_county(feat):
        center = feat.geometry().centroid(maxError=1)
        joined = adm2.filterBounds(center)
        first = ee.Feature(joined.first())
        # 若中心点未落在任何县内（边界附近），用最近的 ADM2
        return feat.set({
            "county_code": ee.Algorithms.If(
                first, first.get("ADM2_CODE"), ee.Number(-1)
            ),
            "county_name": ee.Algorithms.If(
                first, first.get("ADM2_NAME"), ee.String("UNKNOWN")
            ),
        })

    return grid_fc.map(add_county)


def filter_urban_grids(grid_fc, ghsl_collection: str, min_smod: int,
                       min_urban_frac: float = 0.1, year_ref: int = 2020):
    """
    用 GHSL SMOD 过滤：保留至少 min_urban_frac 比例像素 SMOD>=min_smod 的网格。

    SMOD 等级：10=水, 11/12=低密度农村, 13=乡村集群, 21=郊区, 22=半密集城市,
              23=密集城市, 30=城市中心
    """
    smod = (ee.ImageCollection(ghsl_collection)
            .filter(ee.Filter.calendarRange(year_ref, year_ref, "year"))
            .first())
    smod = ee.Image(ee.Algorithms.If(
        smod, smod,
        ee.ImageCollection(ghsl_collection).mosaic()
    )).select("smod_code")

    urban_mask = smod.gte(min_smod)  # binary

    def add_urban_frac(feat):
        # 二值掩膜的 mean = 城镇化像素比例
        frac = urban_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feat.geometry(),
            scale=100,                # GHSL 100m
            maxPixels=1e8,
            bestEffort=True,
        ).values().get(0)
        return feat.set("urban_frac", ee.Number(ee.Algorithms.If(frac, frac, 0)))

    return grid_fc.map(add_urban_frac).filter(
        ee.Filter.gte("urban_frac", min_urban_frac)
    )


# -----------------------
# Sentinel-2 合成
# -----------------------
def mask_scl(img, scl_exclude):
    scl = img.select("SCL")
    mask = ee.Image.constant(1)
    for v in scl_exclude:
        mask = mask.And(scl.neq(v))
    return img.updateMask(mask)


def annual_s2_composite(geom, year: int, cfg: dict):
    months = cfg["data"]["composite_months"]
    start = ee.Date.fromYMD(year, months[0], 1)
    end = ee.Date.fromYMD(year, months[-1] + 1, 1) if months[-1] < 12 else ee.Date.fromYMD(year + 1, 1, 1)

    col = (ee.ImageCollection(cfg["data"]["s2_collection"])
           .filterBounds(geom)
           .filterDate(start, end)
           .map(lambda im: mask_scl(im, cfg["data"]["scl_exclude"])))

    composite = col.median().select(cfg["data"]["bands"])
    return composite


# -----------------------
# VIIRS
# -----------------------
def viirs_annual_mean(grid_fc, year: int, cfg: dict):
    """对每个网格算 year 年的 VIIRS 平均亮度，返回 FeatureCollection。"""
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)
    viirs = (ee.ImageCollection(cfg["data"]["viirs_collection"])
             .filterDate(start, end)
             .select("avg_rad")
             .mean())

    def add_mean(feat):
        v = viirs.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feat.geometry(),
            scale=500,
            maxPixels=1e8,
            bestEffort=True,
        ).get("avg_rad")
        return feat.set({"year": year, "viirs_mean": v})

    return grid_fc.map(add_mean)


# -----------------------
# 任务管理
# -----------------------
def get_existing_task_descriptions() -> set:
    """
    扫已有的 GEE 任务，返回所有处于 READY/RUNNING/COMPLETED 状态的 description 集合。
    用于 --skip-existing：中断后重启时跳过已经提交（或完成）的任务，避免 Drive 文件重复。
    """
    print("[skip-existing] 扫 GEE 任务列表 ...")
    tasks = with_retry(lambda: ee.batch.Task.list(), what="Task.list")
    existing = set()
    for t in tasks[:10000]:
        s = t.status()  # 已经缓存在 task 对象中，不再触发 API
        state = s.get("state", "")
        if state in ("READY", "RUNNING", "COMPLETED"):
            existing.add(s.get("description", ""))
    print(f"     ✓ 已有 {len(existing)} 个 task（未失败）")
    return existing


def wait_if_quota_full(max_pending: int = 2500, check_every_s: int = 60):
    """GEE 限制约 3000 个并发 batch task。若 READY+RUNNING 超过 max_pending 就等到下降。"""
    while True:
        tasks = with_retry(lambda: ee.batch.Task.list(), what="Task.list (throttle)")
        pending = 0
        for t in tasks[:5000]:
            # task.status() 在已经从 list() 拉过的 task 上是缓存数据，不再触发请求
            st = t.status().get("state", "")
            if st in ("READY", "RUNNING"):
                pending += 1
        if pending < max_pending:
            return
        print(f"     [节流] 当前 pending={pending} ≥ {max_pending}，等 {check_every_s}s ...")
        time.sleep(check_every_s)


def submit_export(image, region, file_prefix, drive_folder, scale):
    def _do():
        task = ee.batch.Export.image.toDrive(
            image=image.clip(region).toFloat(),
            description=file_prefix,
            folder=drive_folder,
            fileNamePrefix=file_prefix,
            region=region,
            scale=scale,
            maxPixels=1e10,
            fileFormat="GeoTIFF",
        )
        task.start()
        return task
    return with_retry(_do, what=f"submit {file_prefix}")


def submit_table_export(fc, description, drive_folder):
    def _do():
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=description,
            folder=drive_folder,
            fileNamePrefix=description,
            fileFormat="CSV",
        )
        task.start()
        return task
    return with_retry(_do, what=f"submit table {description}")


def monitor_tasks(task_records, interval_s, log_csv):
    """轮询打印状态，全部完成或失败后返回。"""
    print(f"\n[监控] 开始监控 {len(task_records)} 个任务（每 {interval_s}s 轮询）...")
    while True:
        states = {"READY": 0, "RUNNING": 0, "COMPLETED": 0, "FAILED": 0, "CANCELLED": 0}
        for rec in task_records:
            t = rec["task"]
            try:
                status = t.status()
                state = status.get("state", "UNKNOWN")
                rec["state"] = state
                rec["error"] = status.get("error_message", "")
                states[state] = states.get(state, 0) + 1
            except Exception as e:
                rec["state"] = "ERROR_POLLING"
                rec["error"] = str(e)

        print(f"[监控] {time.strftime('%H:%M:%S')} | " +
              " | ".join(f"{k}={v}" for k, v in states.items() if v > 0))

        # 写日志
        with open(log_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["description", "state", "error"])
            w.writeheader()
            for rec in task_records:
                w.writerow({
                    "description": rec["description"],
                    "state": rec.get("state", ""),
                    "error": rec.get("error", ""),
                })

        # 退出条件：没有 READY 和 RUNNING
        if states.get("READY", 0) == 0 and states.get("RUNNING", 0) == 0:
            break
        time.sleep(interval_s)

    print(f"\n[完成] 任务日志已写入 {log_csv}")


# -----------------------
# 主流程
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--years", default=None, help="例如 2015-2023 / 2020 / 2015,2017")
    ap.add_argument("--max-tasks", type=int, default=None, help="最多提交多少任务（含 S2 + VIIRS + meta），用于冒烟测试")
    ap.add_argument("--dry-run", action="store_true", help="只构建网格和任务，不真正提交")
    ap.add_argument("--no-monitor", action="store_true", help="提交完任务立刻退出（不阻塞监控）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="启动时扫已提交/完成的 GEE 任务，跳过同名任务（断点续跑）")
    ap.add_argument("--project", default=None, help="GEE Cloud Project ID，覆盖 config / env var")
    args = ap.parse_args()

    cfg = load_config(args.config)
    years = parse_years(args.years) if args.years else cfg["data"]["years"]
    max_tasks = args.max_tasks

    project = args.project or cfg["data"]["gee"].get("project")
    init_ee(project=project)

    print(f"[1/5] 加载广东省边界 ...")
    province = get_guangdong_geometry(
        cfg["data"]["region_admin1_name"],
        region_subset=cfg["data"].get("region_subset"),
    )

    print(f"[2/5] 生成 {cfg['data']['grid_size_m']}m 网格 ...")
    grid = build_grid(province, cfg["data"]["grid_size_m"])

    print(f"[3/5] 用 GHSL SMOD>={cfg['data']['ghsl_min_smod']} 且 urban_frac>={cfg['data']['ghsl_min_urban_frac']} 过滤网格 ...")
    grid = filter_urban_grids(
        grid,
        cfg["data"]["ghsl_collection"],
        cfg["data"]["ghsl_min_smod"],
        min_urban_frac=cfg["data"]["ghsl_min_urban_frac"],
    )

    print(f"[3.5/5] 给每个网格分配 county_code（ADM2） ...")
    grid = assign_county(grid, cfg["data"]["region_admin1_name"])
    # 剔除 county 未匹配的网格（中心点落在边界外/海上）
    grid = grid.filter(ee.Filter.neq("county_name", "UNKNOWN"))

    n_grids = grid.size().getInfo()
    print(f"     => 保留 {n_grids} 个网格")

    # 导出网格元数据 CSV
    meta_fc = grid.map(lambda f: ee.Feature(None, {
        "grid_id": f.get("grid_id"),
        "center_lng": f.get("center_lng"),
        "center_lat": f.get("center_lat"),
        "county_code": f.get("county_code"),
        "county_name": f.get("county_name"),
    }))

    drive_folder = cfg["data"]["gee"]["drive_folder"]
    task_records = []

    # 断点续跑：拉一次现有任务列表
    existing = get_existing_task_descriptions() if args.skip_existing else set()
    skipped = 0

    if args.dry_run:
        print(f"\n[Dry-run] 跳过任务提交。将会提交：")
        print(f"   - 1 个 grid_meta.csv")
        print(f"   - {len(years)} 个 VIIRS CSV")
        print(f"   - {n_grids * len(years)} 个 S2 GeoTIFF")
        return

    print(f"[4/5] 提交 grid_meta 表格导出 ...")
    if "grid_meta" in existing:
        print("     [skip-existing] grid_meta 已存在，跳过")
        skipped += 1
    else:
        t = submit_table_export(meta_fc, "grid_meta", drive_folder)
        task_records.append({"task": t, "description": "grid_meta"})

    # 预拉取所有 grid 的元信息（一次批量调用），避免循环中每次都 .getInfo()
    print(f"[5/5a] 预拉取 {n_grids} 个 grid 的元信息（5 个 aggregate_array 调用）...")

    def with_bbox(f):
        c = ee.List(f.geometry().bounds().coordinates().get(0))
        sw = ee.List(c.get(0)); ne = ee.List(c.get(2))
        return f.set({"_x0": sw.get(0), "_y0": sw.get(1),
                      "_x1": ne.get(0), "_y1": ne.get(1)})

    grid_bbox = grid.map(with_bbox)
    grid_ids_local = grid_bbox.aggregate_array("grid_id").getInfo()
    x0s = grid_bbox.aggregate_array("_x0").getInfo()
    y0s = grid_bbox.aggregate_array("_y0").getInfo()
    x1s = grid_bbox.aggregate_array("_x1").getInfo()
    y1s = grid_bbox.aggregate_array("_y1").getInfo()
    print(f"     ✓ 拉到 {len(grid_ids_local)} 条记录")

    print(f"[5/5b] 提交 S2 和 VIIRS 导出任务（按年）...")
    submitted = 1  # 已提交一个 meta 任务

    for year in years:
        # VIIRS：每年一个 CSV
        viirs_desc = f"viirs_{year}"
        if viirs_desc in existing:
            print(f"     [skip-existing] {viirs_desc} 已存在，跳过")
            skipped += 1
        else:
            viirs_fc = viirs_annual_mean(grid, year, cfg)
            viirs_fc_clean = viirs_fc.map(lambda f: ee.Feature(None, {
                "grid_id": f.get("grid_id"),
                "year": f.get("year"),
                "viirs_mean": f.get("viirs_mean"),
            }))
            t = submit_table_export(viirs_fc_clean, viirs_desc, drive_folder)
            task_records.append({"task": t, "description": viirs_desc})
            submitted += 1
            if max_tasks and submitted >= max_tasks:
                break

        # S2：每个网格一个 GeoTIFF —— 纯 Python 端循环，零 round trip
        throttle_every = cfg["data"]["gee"].get("throttle_every", 500)
        max_pending = cfg["data"]["gee"].get("max_pending_tasks", 2500)
        for k, grid_id in enumerate(grid_ids_local):
            if max_tasks and submitted >= max_tasks:
                break
            file_prefix = f"{grid_id}_{year}"
            if file_prefix in existing:
                skipped += 1
                continue
            # 每提交 throttle_every 个任务检查一次 quota，避免突破 GEE 上限
            if submitted > 0 and submitted % throttle_every == 0:
                wait_if_quota_full(max_pending=max_pending)
            geom = ee.Geometry.Rectangle([x0s[k], y0s[k], x1s[k], y1s[k]])
            composite = annual_s2_composite(geom, year, cfg)
            t = submit_export(
                composite, geom, file_prefix, drive_folder, cfg["data"]["s2_scale"]
            )
            task_records.append({"task": t, "description": file_prefix})
            submitted += 1
            if submitted % 100 == 0:
                print(f"     ... 已提交 {submitted} 个，跳过 {skipped} 个")

        if max_tasks and submitted >= max_tasks:
            print(f"[限制] 达到 --max-tasks={max_tasks}，停止提交。")
            break

    print(f"\n[汇总] 共提交 {len(task_records)} 个任务到 Drive/{drive_folder}")

    log_csv = cfg["data"]["paths"]["task_log_csv"]
    Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
    if args.no_monitor:
        print(f"[--no-monitor] 跳过任务监控。可在 https://code.earthengine.google.com/tasks 查看进度。")
        # 至少写一次初始状态
        with open(log_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["description", "state", "error"])
            w.writeheader()
            for rec in task_records:
                w.writerow({"description": rec["description"],
                            "state": "SUBMITTED", "error": ""})
    else:
        monitor_tasks(
            task_records,
            cfg["data"]["gee"]["monitor_interval_s"],
            log_csv,
        )

    print(f"\n[下一步] 从 Google Drive 文件夹 '{drive_folder}' 下载所有产物：")
    print(f"   - GeoTIFF → {cfg['data']['paths']['sentinel2_dir']}/")
    print(f"   - grid_meta.csv → {cfg['data']['paths']['grid_meta_csv']}")
    print(f"   - viirs_YYYY.csv 合并后 → {cfg['data']['paths']['viirs_csv']}")


if __name__ == "__main__":
    main()
