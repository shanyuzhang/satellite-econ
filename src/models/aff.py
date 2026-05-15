"""
AFF — Attentional Feature Fusion （Dai et al., 2021；Cheng & Jiang 2024 用它融合卫星图像与能源数据）

本实现是 1D 版（输入是已 pool 的特征向量），适合图像 backbone 输出的 (B, dim_img) 与
数值特征经线性投影后的 (B, dim_num) 融合。

接口：
    aff = AFF(dim_x, dim_y, mid_dim, reduction=4)
    out = aff(x, y)         # x: (B, dim_x), y: (B, dim_y) -> (B, mid_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AFF(nn.Module):
    def __init__(self, dim_x: int, dim_y: int, mid_dim: int = 256, reduction: int = 4):
        super().__init__()
        self.proj_x = nn.Linear(dim_x, mid_dim)
        self.proj_y = nn.Linear(dim_y, mid_dim)

        bottleneck = max(mid_dim // reduction, 4)

        # 局部注意力（在特征维上的 1×1 mlp，模拟原 AFF 的 PWConv 路径）
        self.local_att = nn.Sequential(
            nn.Linear(mid_dim, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, mid_dim),
            nn.BatchNorm1d(mid_dim),
        )

        # 全局注意力：对 1D 特征向量，BN 与 local_att 等价；这里用独立分支学习不同权重
        self.global_att = nn.Sequential(
            nn.Linear(mid_dim, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, mid_dim),
            nn.BatchNorm1d(mid_dim),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xp = self.proj_x(x)
        yp = self.proj_y(y)

        # AFF 原文：先求和后过注意力 → 得到融合权重
        xy = xp + yp
        att_l = self.local_att(xy)
        att_g = self.global_att(xy)
        w = self.sigmoid(att_l + att_g)

        # 加权融合：2*x*w + 2*y*(1-w)（保持输出量级与单分支相当）
        return 2.0 * xp * w + 2.0 * yp * (1.0 - w)
