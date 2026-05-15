"""
M3：纯 ViT-Base 基线（timm，9 通道，ImageNet 预训练）+ AFF。

目的：与 M2 对比，分离"提升来自 Transformer"还是"来自混合架构"。
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .aff import AFF


class ViTBaseline(nn.Module):
    def __init__(
        self,
        in_chans: int = 9,
        num_features: int = 2,
        aff_mid: int = 256,
        aff_reduction: int = 4,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224",
    ):
        super().__init__()
        # timm 的 in_chans!=3 时会自动复制 RGB 预训练权重
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,           # 去掉分类头，输出全局特征
            global_pool="token",     # 用 CLS token
        )
        self.feat_dim = self.backbone.num_features  # 768

        self.aff = AFF(dim_x=self.feat_dim, dim_y=num_features,
                       mid_dim=aff_mid, reduction=aff_reduction)
        self.head = nn.Sequential(
            nn.Linear(aff_mid, aff_mid // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(aff_mid // 2, 1),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        f = self.backbone(image)             # (B, 768)
        fused = self.aff(f, features)
        return self.head(fused)

    def param_groups(self, lr_cfg: dict) -> list:
        head_params = list(self.aff.parameters()) + list(self.head.parameters())
        backbone_params = list(self.backbone.parameters())
        return [
            {"params": backbone_params, "lr": lr_cfg["backbone"], "name": "backbone"},
            {"params": head_params, "lr": lr_cfg["head"], "name": "head"},
        ]
