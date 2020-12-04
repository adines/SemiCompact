# SOTA-tinyML

## Networks:
* SqueezeNet: [Available at Pytorch](https://pytorch.org/docs/stable/torchvision/models.html) 
* ShuffleNet v2: [Available at Pytorch](https://pytorch.org/docs/stable/torchvision/models.html)
* MobileNet v2: [Available at Pytorch](https://pytorch.org/docs/stable/torchvision/models.html)
* MNasNet: [Available at Pytorch](https://pytorch.org/docs/stable/torchvision/models.html)
* FBNetsV2: [Available at Facebook GitHub repository](https://github.com/facebookresearch/mobile-vision)
* MixNet: [Pytorch implementation](https://github.com/ansleliu/MixNet-PyTorch)
* Single-Path Nas


## Results
### SIIM-ISIC Melanoma
Accuracy

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 83.0 | - | - | - | - | - | - |
| ResNet101 | 83.0 | - | - | - | - | - | - |
| EfficientNet | 84.0 | - | - | - | - | - | - |
| FBNet | 58.5 | 82.5 | **84.5** | 83.5 | 82.5 | **84.5** | 78.0 |
| MixNet | 50.0 | 75.5 | 67.5 | 69.5 | 74.5 | 72.0 | - |
| MNasNet | 53.0 | 80.5 | 79.5 | 50.0 | 50.0 | 73.5 | 68.5 |
| MobileNet | 82.0 | 80.5 | 82.5 | 77.0 | 78.5 | 81.5 | 77.5 |
| ResNet18 | 81.5 | 82.0 | 81.0 | **84.5** | 81.5 | 82.0 | 75.0 |
| SqeezeNet | 78.5 | 78.5 | 80.5 | 80.5 | 77.5 | 77.0 | - |
| ShuffleNet | 78.5 | 77.5 | 78.0 | 77.5 | 77.5 | 78.5 | - |

F1-Score

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 82.8 | - | - | - | - | - | - |
| ResNet101 | 82.1 | - | - | - | - | - | - |
| EfficientNet | 82.8 | - | - | - | - | - | - |
| FBNet | 36.6 | 81.7 | 83.6 | 82.7 | 80.4 | 84.3 | 79.2 |
| MixNet | 0 | 71.7 | 55.2 | 62.1 | 68.7 | 74.8 | - |
| MNasNet | 19.0 | 78.2 | 76.0 | 50.0 | 50.0 | 69.0 | 74.3 |
| MobileNet | 80.4 | 78.0 | 81.3 | 70.9 | 74.3 | 80.2 | 76.4 |
| ResNet18 | 81.4 | 82.2 | 80.4 | **84.6** | 80.8 | 82.0 | 77.9 |
| SqeezeNet | 78.2 | 75.1 | 78.7 | 77.2 | 73.7 | 77.2 | - |
| ShuffleNet | 80.5 | 76.7 | 77.3 | 74.6 | 73.4 | 81.1 | - |

Precision

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 83.7 | - | - | - | - | - | - |
| ResNet101 | 86.7 | - | - | - | - | - | - |
| EfficientNet | 89.5 | - | - | - | - | - | - |
| FBNet | 77.4 | 85.7 | 88.8 | 86.8 | 91.1 | 85.6 | 75.0 |
| MixNet | 0 | 84.9 | 88.9 | 82.0 | 88.9 | 68.0 | - |
| MNasNet | 68.8 | 88.6 | 91.5 | 50.0 | 50.0 | 83.1 | 62.8 |
| MobileNet | 88.1 | 89.6 | 87.4 | **96.6** | 92.5 | 86.2 | 80.2 |
| ResNet18 | 81.8 | 81.4 | 83.0 | 84.2 | 83.9 | 82.0 | 69.8 |
| SqeezeNet | 79.4 | 89.0 | 86.7 | 93.0 | 88.7 | 76.5 | - |
| ShuffleNet | 73.6 | 79.6 | 79.8 | 85.7 | 89.9 | 72.4 | - |

Recall

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 82.0 | - | - | - | - | - | - |
| ResNet101 | 78.0 | - | - | - | - | - | - |
| EfficientNet | 77.0 | - | - | - | - | - | - |
| FBNet | 24.0 | 78.0 | 79.0 | 79.0 | 72.0 | 83.0 | 84.0 |
| MixNet | 0 | 62.0 | 40.0 | 50.0 | 56.0 | 83.0 | - |
| MNasNet | 11.0 | 70.0 | 65.0 | 50.0 | 50.0 | 59.0 | 91.0 |
| MobileNet | 74.0 | 69.0 | 76.0 | 56.0 | 62.0 | 75.0 | 73.0 |
| ResNet18 | 81.0 | 83.0 | 78.0 | 85.0 | 78.0 | 82.0 | 88.0 |
| SqeezeNet | 77.0 | 65.0 | 72.0 | 66.0 | 63.0 | 78.0 | - |
| ShuffleNet | 89.0 | 74.0 | 75.0 | 66.0 | 62.0 | **92.0** | - |

### APTOS Blindness

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 82.1 | - | - | - | - | - | - |
| ResNet101 | 83.0 | - | - | - | - | - | - |
| EfficientNet | 82.1 | - | - | - | - | - | - |
| FBNet | 83.0 | 83.0 | 83.3 | 83.3 | 83.7 | 83.6 | 78.5 |
| MixNet | 72.2 | 71.1 | 70.3 | 71.3 | 70.9 | 72.64 | - |
| MNasNet | 70.7 | 80.0 | 80.0 | 64.4 | 78.3 | 78.2 | 80.8 |
| MobileNet | 80.4 | 81.5 | 82.8 | 81.8 | 79.3 | **84.8** | - |
| ResNet18 | 83.3 | 82.5 | 82.1 | 82.5 | 82.9 | 84.7 | 70.5 |
| SqeezeNet | 78.8 | 82.1 | 79.5 | 81.0 | 81.0 | 82.2 | - |
| ShuffleNet | 72.9 | 72.8 | 72.8 | 73.9 | 72.4 | 72.6 | - |
