"""模型工厂：根据 config 返回 M1/M2/M3/M4 实例。"""

from __future__ import annotations

from .aff import AFF
from .cnn_transformer import CNNTransformer
from .resnet_baseline import ResNetBaseline
from .satmae_finetune import SatMAEFinetune
from .vit_baseline import ViTBaseline


def build_model(name: str, num_features: int, config: dict):
    """
    Args:
        name: M1 | M2 | M3 | M4
        num_features: 数值特征维度
        config: 完整 config dict
    """
    in_chans = len(config["data"]["bands"])
    aff_cfg = config["model"]["aff"]
    aff_mid = aff_cfg["mid_dim"]
    aff_red = aff_cfg["reduction"]

    if name == "M1":
        return ResNetBaseline(
            in_chans=in_chans, num_features=num_features,
            aff_mid=aff_mid, aff_reduction=aff_red, pretrained=True,
        )
    if name == "M2":
        tcfg = config["model"]["transformer"]
        return CNNTransformer(
            in_chans=in_chans, num_features=num_features,
            d_model=tcfg["dim"], nhead=tcfg["heads"], num_layers=tcfg["layers"],
            mlp_ratio=tcfg["mlp_ratio"], dropout=tcfg["dropout"],
            aff_mid=aff_mid, aff_reduction=aff_red, pretrained=True,
        )
    if name == "M3":
        return ViTBaseline(
            in_chans=in_chans, num_features=num_features,
            aff_mid=aff_mid, aff_reduction=aff_red, pretrained=True,
        )
    if name == "M4":
        return SatMAEFinetune(
            in_chans=in_chans, num_features=num_features,
            aff_mid=aff_mid, aff_reduction=aff_red,
            weights_path=config["model"]["satmae"]["weights_path"],
            freeze_layers=config["train"]["lr"]["M4"]["freeze_layers"],
        )
    raise ValueError(f"未知模型 name={name}（必须是 M1/M2/M3/M4）")


__all__ = ["build_model", "AFF", "ResNetBaseline", "CNNTransformer",
           "ViTBaseline", "SatMAEFinetune"]
