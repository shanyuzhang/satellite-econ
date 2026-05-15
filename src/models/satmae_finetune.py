"""
M4：SatMAE / ViT-Large 微调 + AFF。

加载策略：
  1. 用 timm 的 vit_large_patch16_224 创建 9 通道模型（ImageNet 权重）。
  2. 若 `weights/satmae_vit_large.pth` 存在，覆盖加载 SatMAE 预训练 state_dict。
  3. 冻结前 freeze_layers 个 Transformer block，只微调后几个 block + 回归头。

SatMAE 项目主页：https://github.com/sustainlab-group/SatMAE
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn

from .aff import AFF


def _load_satmae_state_dict(model: nn.Module, weights_path: str) -> dict:
    """
    把 SatMAE checkpoint load 到 timm 的 ViT-Large 上。
    SatMAE 可能用 'model' 或 'state_dict' 子键 + 'module.' 前缀，需要清理。
    返回 missing/unexpected 报告。
    """
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    cleaned = {}
    for k, v in ckpt.items():
        nk = k.replace("module.", "").replace("backbone.", "")
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[SatMAE] 加载完成。missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"[SatMAE] missing keys (前 10): {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[SatMAE] unexpected keys (前 10): {unexpected[:10]}")
    return {"missing": missing, "unexpected": unexpected}


class SatMAEFinetune(nn.Module):
    def __init__(
        self,
        in_chans: int = 9,
        num_features: int = 2,
        aff_mid: int = 256,
        aff_reduction: int = 4,
        weights_path: Optional[str] = "weights/satmae_vit_large.pth",
        freeze_layers: int = 20,
    ):
        super().__init__()

        # 先用 timm 创建 ViT-Large（自动复制 RGB → 9 通道）
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=True,
            in_chans=in_chans,
            num_classes=0,
            global_pool="token",
        )
        self.feat_dim = self.backbone.num_features  # 1024

        # 尝试加载 SatMAE 权重
        if weights_path and Path(weights_path).exists():
            print(f"[SatMAE] 找到权重文件 {weights_path}，尝试加载...")
            _load_satmae_state_dict(self.backbone, weights_path)
        else:
            print(f"[SatMAE] 未找到 {weights_path}，回退到 timm 的 ImageNet 预训练。"
                  "如需 SatMAE 请参照 weights/README.md 手动下载。")

        # 冻结前 freeze_layers 个 block
        self._freeze_n_blocks(freeze_layers)

        self.aff = AFF(dim_x=self.feat_dim, dim_y=num_features,
                       mid_dim=aff_mid, reduction=aff_reduction)
        self.head = nn.Sequential(
            nn.Linear(aff_mid, aff_mid // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(aff_mid // 2, 1),
        )

    def _freeze_n_blocks(self, n: int):
        # 冻结 patch_embed + 前 n 个 block
        if hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token.requires_grad = False
        if hasattr(self.backbone, "pos_embed"):
            self.backbone.pos_embed.requires_grad = False
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            print("[SatMAE] backbone 没有 .blocks 属性，跳过冻结。")
            return
        n = min(n, len(blocks))
        for i in range(n):
            for p in blocks[i].parameters():
                p.requires_grad = False
        print(f"[SatMAE] 冻结 patch_embed + cls_token + pos_embed + 前 {n}/{len(blocks)} 个 block。")

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        f = self.backbone(image)
        fused = self.aff(f, features)
        return self.head(fused)

    def param_groups(self, lr_cfg: dict) -> list:
        # 把可训练参数分为两组：backbone 中没被冻结的 + head
        finetune_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.aff.parameters()) + list(self.head.parameters())
        return [
            {"params": finetune_params, "lr": lr_cfg["finetune"], "name": "finetune"},
            {"params": head_params, "lr": lr_cfg["head"], "name": "head"},
        ]
