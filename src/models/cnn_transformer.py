"""
M2：CNN→Transformer 混合（主模型）

- ResNet-50 前 3 个 stage 抽取局部特征：输入 (B,9,224,224) → (B,1024,14,14)
- 展平为 196 个 token，线性投影到 d=256，加可学习位置编码和 [CLS] token
- 4 层 Transformer Encoder（8 heads, GELU, batch_first）
- [CLS] 输出经 AFF 融合数值特征 → FC → 预测值
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .aff import AFF
from .resnet_baseline import inflate_conv1


class CNNTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int = 9,
        num_features: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        aff_mid: int = 256,
        aff_reduction: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        resnet.conv1 = inflate_conv1(resnet.conv1, in_chans)

        # 截取到 layer3：输出 (B, 1024, 14, 14) when input is 224
        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        self.cnn_out_dim = 1024
        self.num_tokens = 14 * 14  # 196

        self.token_proj = nn.Linear(self.cnn_out_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.d_model = d_model

        self.aff = AFF(dim_x=d_model, dim_y=num_features,
                       mid_dim=aff_mid, reduction=aff_reduction)
        self.head = nn.Sequential(
            nn.Linear(aff_mid, aff_mid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aff_mid // 2, 1),
        )

        # 用于可视化时的钩子
        self._attn_weights = None

    # ----------- forward -----------
    def _tokenize(self, image: torch.Tensor) -> torch.Tensor:
        f = self.cnn(image)              # (B, 1024, 14, 14)
        B, C, H, W = f.shape
        tokens = f.flatten(2).transpose(1, 2)        # (B, 196, 1024)
        tokens = self.token_proj(tokens)             # (B, 196, d)
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, 197, d)
        tokens = tokens + self.pos_embed
        return tokens

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        tokens = self._tokenize(image)
        enc = self.transformer(tokens)               # (B, 197, d)
        enc = self.norm(enc)
        cls_out = enc[:, 0]                          # (B, d)
        fused = self.aff(cls_out, features)
        return self.head(fused)

    @torch.no_grad()
    def forward_with_attention(self, image: torch.Tensor, features: torch.Tensor):
        """
        返回 (pred, attn_map)，attn_map: (B, 14, 14)，是最后一层 CLS→patch 的注意力。
        实现：手动跑 transformer，最后一层捕获 attn weights。
        """
        tokens = self._tokenize(image)
        x = tokens
        last_attn = None
        for i, layer in enumerate(self.transformer.layers):
            is_last = i == len(self.transformer.layers) - 1
            # 复制 nn.TransformerEncoderLayer 的 norm_first=True 前向逻辑
            normed = layer.norm1(x)
            attn_out, attn_w = layer.self_attn(
                normed, normed, normed,
                need_weights=is_last, average_attn_weights=True,
            )
            x = x + layer.dropout1(attn_out)
            x = x + layer._ff_block(layer.norm2(x))
            if is_last:
                last_attn = attn_w  # (B, 197, 197)

        x = self.norm(x)
        cls_out = x[:, 0]
        pred = self.head(self.aff(cls_out, features))

        # CLS 对 patch tokens 的 attention：取第 0 行（CLS query）对 1..196 个 patch key
        cls_to_patch = last_attn[:, 0, 1:]                    # (B, 196)
        attn_map = cls_to_patch.reshape(-1, 14, 14)            # (B, 14, 14)
        return pred, attn_map

    # ----------- param groups -----------
    def param_groups(self, lr_cfg: dict) -> list:
        cnn_params = list(self.cnn.parameters())
        transformer_params = (
            [self.cls_token, self.pos_embed]
            + list(self.token_proj.parameters())
            + list(self.transformer.parameters())
            + list(self.norm.parameters())
        )
        head_params = list(self.aff.parameters()) + list(self.head.parameters())
        return [
            {"params": cnn_params, "lr": lr_cfg["cnn"], "name": "cnn"},
            {"params": transformer_params, "lr": lr_cfg["transformer"], "name": "transformer"},
            {"params": head_params, "lr": lr_cfg["head"], "name": "head"},
        ]
