"""
M1：ResNet-50 (9 通道) + AFF 融合数值特征。

输入：
    image:    (B, 9, 224, 224)
    features: (B, num_features)
输出：
    pred:     (B, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .aff import AFF


def inflate_conv1(conv1_rgb: nn.Conv2d, in_chans: int) -> nn.Conv2d:
    """
    把 ResNet 的 RGB conv1 扩展到 in_chans 通道：
    - 前 3 个通道复用 RGB 预训练权重
    - 多出来的通道用 RGB 通道均值初始化（保留预训练信号，比随机初始化好）
    """
    assert conv1_rgb.in_channels == 3, "conv1_rgb 必须是 3 通道"
    if in_chans == 3:
        return conv1_rgb

    new_conv = nn.Conv2d(
        in_channels=in_chans,
        out_channels=conv1_rgb.out_channels,
        kernel_size=conv1_rgb.kernel_size,
        stride=conv1_rgb.stride,
        padding=conv1_rgb.padding,
        bias=conv1_rgb.bias is not None,
    )
    with torch.no_grad():
        w = conv1_rgb.weight  # (out, 3, k, k)
        mean_w = w.mean(dim=1, keepdim=True)  # (out, 1, k, k)
        new_w = torch.zeros(conv1_rgb.out_channels, in_chans,
                            w.shape[2], w.shape[3], dtype=w.dtype)
        new_w[:, :3] = w
        for i in range(3, in_chans):
            new_w[:, i:i + 1] = mean_w
        new_conv.weight.copy_(new_w)
        if conv1_rgb.bias is not None:
            new_conv.bias.copy_(conv1_rgb.bias)
    return new_conv


class ResNetBaseline(nn.Module):
    """M1：完整 ResNet-50，GAP 后过 AFF 融合数值特征，再 FC 出预测。"""

    def __init__(self, in_chans: int = 9, num_features: int = 2,
                 aff_mid: int = 256, aff_reduction: int = 4, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        resnet.conv1 = inflate_conv1(resnet.conv1, in_chans)

        # 把分类头扔掉，保留到 avgpool
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
        )
        self.feat_dim = 2048

        self.aff = AFF(dim_x=self.feat_dim, dim_y=num_features,
                       mid_dim=aff_mid, reduction=aff_reduction)
        self.head = nn.Sequential(
            nn.Linear(aff_mid, aff_mid // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(aff_mid // 2, 1),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        f = self.backbone(image).flatten(1)  # (B, 2048)
        fused = self.aff(f, features)        # (B, aff_mid)
        return self.head(fused)              # (B, 1)

    # 用于 train.py 构 param groups
    def param_groups(self, lr_cfg: dict) -> list:
        head_params = list(self.aff.parameters()) + list(self.head.parameters())
        backbone_params = list(self.backbone.parameters())
        return [
            {"params": backbone_params, "lr": lr_cfg["backbone"], "name": "backbone"},
            {"params": head_params, "lr": lr_cfg["head"], "name": "head"},
        ]
