# 模型权重目录

## SatMAE 权重（M4 使用）

M4 模型默认尝试加载 SatMAE 在 Sentinel-2 上预训练的 ViT-Large 权重。如果文件不存在，
模型会自动回退到 timm 的 ImageNet 预训练 ViT-Large。

### 手动下载步骤

1. 访问 SatMAE 官方仓库：https://github.com/sustainlab-group/SatMAE
2. 找到 Sentinel-2 预训练的 ViT-Large checkpoint（README 中的 "Pretrained Weights" 部分）。
3. 下载后重命名为 `satmae_vit_large.pth`，放到本目录：

```
weights/satmae_vit_large.pth
```

4. 如果权重的 key 命名与 timm 的 `vit_large_patch16_224` 不完全一致，
   `src/models/satmae_finetune.py` 里的 `_load_satmae_state_dict()` 会做基本的前缀
   映射；如果仍有 mismatch，请按需修改该函数。

### 路径覆盖

也可以在训练命令里通过 config 或 CLI 覆盖路径（见 `configs/default.yaml` 的
`model.satmae.weights_path`）。

## 其他权重

ResNet-50 / ViT-Base / ViT-Large 的 ImageNet 权重由 torchvision / timm 自动下载到
HuggingFace / torch hub 缓存目录，无需手动放置。
