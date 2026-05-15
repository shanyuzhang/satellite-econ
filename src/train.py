"""
训练入口。

用法：
    python -m src.train --model M2 --mode level
    python -m src.train --model M1 --mode diff1 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .dataset import get_dataloaders
from .models import build_model
from .utils import get_device, load_config, r2_score, set_seed

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, choices=["M1", "M2", "M3", "M4"],
                    help="覆盖 config 中的 train.model")
    ap.add_argument("--mode", default=None, choices=["level", "diff1"],
                    help="覆盖 config 中的 train.mode")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    return ap.parse_args()


def init_wandb(config: dict, model_name: str, mode: str, run_name: str):
    if not HAS_WANDB:
        print("[wandb] 未安装 wandb，跳过日志上报。")
        return None
    wcfg = config["wandb"]
    if wcfg["mode"] == "disabled":
        print("[wandb] mode=disabled，跳过日志上报。")
        return None
    run = wandb.init(
        project=wcfg["project"],
        entity=wcfg["entity"],
        name=run_name,
        mode=wcfg["mode"],
        config={"model": model_name, "mode": mode, **config},
    )
    return run


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            feat = batch["features"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            pred = model(img, feat).squeeze(-1)
            loss = criterion(pred, y)
            losses.append(loss.item() * y.size(0))
            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    n = sum(p.shape[0] for p in preds)
    mean_loss = sum(losses) / max(n, 1)
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(labels) if labels else np.array([])
    r2 = r2_score(y_true, y_pred) if n > 0 else float("nan")
    return mean_loss, r2


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.model:
        config["train"]["model"] = args.model
    if args.mode:
        config["train"]["mode"] = args.mode
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size

    model_name = config["train"]["model"]
    mode = config["train"]["mode"]
    seed = config["train"]["seed"]
    set_seed(seed)
    device = get_device()
    print(f"[训练] 模型={model_name}  模式={mode}  device={device}")

    # 数据
    loaders = get_dataloaders(config)
    num_features = loaders["num_features"]
    print(f"[数据] num_features={num_features}, target_col={loaders['target_col']}")

    # 模型
    model = build_model(model_name, num_features=num_features, config=config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 参数总数 {n_params/1e6:.2f}M，可训练 {n_trainable/1e6:.2f}M")

    # 优化器
    lr_cfg = config["train"]["lr"][model_name]
    param_groups = model.param_groups(lr_cfg)
    optimizer = AdamW(
        param_groups,
        weight_decay=config["train"]["weight_decay"],
    )

    # 调度器：linear warmup + cosine
    epochs = config["train"]["epochs"]
    warmup = config["train"]["warmup_epochs"]
    steps_per_epoch = max(len(loaders["train"]), 1)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch

    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                            total_iters=max(warmup_steps, 1))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[max(warmup_steps, 1)])

    criterion = nn.MSELoss()

    # 输出准备
    ckpt_dir = Path("outputs/checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/logs"); log_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_name}_{mode}_seed{seed}"
    best_ckpt = ckpt_dir / f"{run_name}_best.pt"
    log_csv = log_dir / f"{run_name}.csv"

    # wandb
    run = init_wandb(config, model_name, mode, run_name)

    # 训练循环
    best_r2 = -float("inf")
    patience = config["train"]["early_stop_patience"]
    epochs_since_improve = 0

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_r2",
                                "valid_loss", "valid_r2", "lr_head"])

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses, tr_preds, tr_labels = [], [], []
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            img = batch["image"].to(device, non_blocking=True)
            feat = batch["features"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(img, feat).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            if config["train"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config["train"]["grad_clip"],
                )
            optimizer.step()
            scheduler.step()

            tr_losses.append(loss.item() * y.size(0))
            tr_preds.append(pred.detach().cpu().numpy())
            tr_labels.append(y.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n_tr = sum(p.shape[0] for p in tr_preds)
        tr_loss = sum(tr_losses) / max(n_tr, 1)
        tr_r2 = r2_score(np.concatenate(tr_labels), np.concatenate(tr_preds)) if n_tr else float("nan")

        val_loss, val_r2 = evaluate(model, loaders["valid"], device, criterion)

        # 取一个代表性学习率（head 组）
        lr_head = None
        for pg in optimizer.param_groups:
            if pg.get("name") == "head":
                lr_head = pg["lr"]
                break

        print(f"epoch {epoch:3d} | tr_loss={tr_loss:.4f} tr_R²={tr_r2:.4f} | "
              f"val_loss={val_loss:.4f} val_R²={val_r2:.4f} | lr_head={lr_head}")

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_r2, val_loss, val_r2, lr_head])

        if run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss, "train/r2": tr_r2,
                "valid/loss": val_loss, "valid/r2": val_r2,
                "lr/head": lr_head,
            })

        # 早停 + 保存最佳
        if (not math.isnan(val_r2)) and val_r2 > best_r2:
            best_r2 = val_r2
            epochs_since_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "model_name": model_name,
                "mode": mode,
                "valid_r2": val_r2,
            }, best_ckpt)
            print(f"  ↳ 新最佳 R²={val_r2:.4f}，已保存到 {best_ckpt}")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"\n[早停] 连续 {patience} epoch 验证集 R² 未提升，停止训练。")
                break

    print(f"\n[完成] 最佳验证集 R² = {best_r2:.4f}")
    print(f"[完成] 检查点：{best_ckpt}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
