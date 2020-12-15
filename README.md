# Semi-Supervised Learning for Image Classification using Compact Networks in the Medical Context
The development of mobile applications that embed deep convolutional neural models 
has the potential to revolutionise healthcare. However, most deep learning models 
require computational resources that are not available in smartphones or edge devices; 
an issue that can be faced by means of compact models. The problem with such models is that 
they are, at least usually, less accurate than bigger models.
In this work, we address this limitation of compact networks with the application of 
semi-supervised learning techniques, which take advantage of unlabelled data. 
Using this combination, we have shown that it is possible to construct compact 
models as accurate as bigger models in two widely employed datasets for [melanoma 
classification](https://www.kaggle.com/c/siim-isic-melanoma-classification) 
and [diabetic retinopathy detection](https://journals.sagepub.com/doi/10.1177/193229680900300315). 
Finally, to facilitate the application of the methods studied in this work, we have developed a 
library that simplifies the construction of compact models using semi-supervised learning methods. 

## Networks
In this work, we explore a variety of both manually and automatically designed architectures. 
Namely, we have employed 4 manually designed compact networks, and 3 automatically desiged 
networks. In addition, for our experiments, we have considered three standard size networks 
that are Resnet-50 and Resnet-101, and EfficientNet-B3. We provide a comparison of different 
features of these networks in the following table.

| Network | Params (M) | FLOPs (M) | Top-1 acc (%) | Top-5 acc (%) | Design | Implementation |
|--|--|--|--|--|--|--|
| ResNet50 | 26 | 4100 | 76.0 | 93.0 | Manual | [Official Pytorch]((https://pytorch.org/docs/stable/torchvision/models.html)) |
| ResNet101 | 44 | 8540 | 80.9 | 95.6 | Manual | [Official Pytorch]((https://pytorch.org/docs/stable/torchvision/models.html)) |
| EfficientNet | 12 | 1800 | 81.6 | 95.7 | Auto | [Unofficial Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) |
| FBNet | 9.4 | 753 | 78.9 | 94.3 | Auto | [Facebook](https://github.com/facebookresearch/mobile-vision)
| MixNet | 5 | 360 | 78.9 | 94.2 | Auto | [Unofficial implementation](https://github.com/ansleliu/MixNet-PyTorch)
| MNasNet | 5.2 | 403 | 75.6 | 92.7 | Auto | [Official Pytorch]((https://pytorch.org/docs/stable/torchvision/models.html)) |
| MobileNet | 3.4 | 300 | 74.7 | 92.5 | Manual | [Official Pytorch]((https://pytorch.org/docs/stable/torchvision/models.html)) |
| ResNet18 | 11 | 1300 | 69.6 | 89.2 | Manual | [Official Pytorch]((https://pytorch.org/docs/stable/torchvision/models.html)) |
| SqeezeNet | 1.3 | 833 | 57.5 | 80.3 | Manual | [Official Pytorch]((https://pytorch.org/docs/stable/torchvision/models.html)) |
| ShuffleNet | 5.3 | 524 | 69.4 | 88.3 | Manual | [Official Pytorch]((https://pytorch.org/docs/stable/torchvision/models.html)) |


## Results
In this section, we present a thorough analysis for the results obtained by the 7 
compact networks and the 6 semi-supervised processes. In particular, we first compare 
the performance of the networks when training using only the labelled data. 
Then, we study the impact of applying the different semi-supervised methods. 

All the networks used in our experiments are implemented in Pytorch, and have been trained 
thanks to the functionality of the Fastai library. In addition, we have used a 
GPU Nvidia RTX 2080 Ti for training the models.

### SIIM-ISIC Melanoma
Performance comparison of the different architectures trained with the seven 
different processes (Base, PD: Plain Distillation, DD: Data Distillation, MD: Model 
Distillation, MDD: Model Data Distillation, FixMatch and MixMatch) in the SIIM-ISIC Melanoma dataset. 
In bold face the best model.

In this case we have used 4 different metrics: Accuracy, F1-score, Precision and Recall;
to compare the performance of the different architectures.
#### Accuracy

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 83.0 | - | - | - | - | - | - |
| ResNet101 | 83.0 | - | - | - | - | - | - |
| EfficientNet | 84.0 | - | - | - | - | - | - |
| FBNet | 58.5 | 82.5 | **84.5** | 83.5 | 82.5 | **84.5** | 78.0 |
| MixNet | 50.0 | 75.5 | 67.5 | 69.5 | 74.5 | 72.0 | 73.5 |
| MNasNet | 53.0 | 80.5 | 79.5 | 50.0 | 50.0 | 73.5 | 68.5 |
| MobileNet | 82.0 | 80.5 | 82.5 | 77.0 | 78.5 | 81.5 | 77.5 |
| ResNet18 | 81.5 | 82.0 | 81.0 | **84.5** | 81.5 | 82.0 | 75.0 |
| SqeezeNet | 78.5 | 78.5 | 80.5 | 80.5 | 77.5 | 77.0 | 83.0 |
| ShuffleNet | 78.5 | 77.5 | 78.0 | 77.5 | 77.5 | 78.5 | 75.0 |

#### F1-Score

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 82.8 | - | - | - | - | - | - |
| ResNet101 | 82.1 | - | - | - | - | - | - |
| EfficientNet | 82.8 | - | - | - | - | - | - |
| FBNet | 36.6 | 81.7 | 83.6 | 82.7 | 80.4 | 84.3 | 79.2 |
| MixNet | 0 | 71.7 | 55.2 | 62.1 | 68.7 | 74.8 | 77.8 |
| MNasNet | 19.0 | 78.2 | 76.0 | 50.0 | 50.0 | 69.0 | 74.3 |
| MobileNet | 80.4 | 78.0 | 81.3 | 70.9 | 74.3 | 80.2 | 76.4 |
| ResNet18 | 81.4 | 82.2 | 80.4 | **84.6** | 80.8 | 82.0 | 77.9 |
| SqeezeNet | 78.2 | 75.1 | 78.7 | 77.2 | 73.7 | 77.2 | 83.7 |
| ShuffleNet | 80.5 | 76.7 | 77.3 | 74.6 | 73.4 | 81.1 | 78.8 |

#### Precision

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 83.7 | - | - | - | - | - | - |
| ResNet101 | 86.7 | - | - | - | - | - | - |
| EfficientNet | 89.5 | - | - | - | - | - | - |
| FBNet | 77.4 | 85.7 | 88.8 | 86.8 | 91.1 | 85.6 | 75.0 |
| MixNet | 0 | 84.9 | 88.9 | 82.0 | 88.9 | 68.0 | 66.9 |
| MNasNet | 68.8 | 88.6 | 91.5 | 50.0 | 50.0 | 83.1 | 62.8 |
| MobileNet | 88.1 | 89.6 | 87.4 | **96.6** | 92.5 | 86.2 | 80.2 |
| ResNet18 | 81.8 | 81.4 | 83.0 | 84.2 | 83.9 | 82.0 | 69.8 |
| SqeezeNet | 79.4 | 89.0 | 86.7 | 93.0 | 88.7 | 76.5 | 80.6 |
| ShuffleNet | 73.6 | 79.6 | 79.8 | 85.7 | 89.9 | 72.4 | 68.4 |

#### Recall

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 82.0 | - | - | - | - | - | - |
| ResNet101 | 78.0 | - | - | - | - | - | - |
| EfficientNet | 77.0 | - | - | - | - | - | - |
| FBNet | 24.0 | 78.0 | 79.0 | 79.0 | 72.0 | 83.0 | 84.0 |
| MixNet | 0 | 62.0 | 40.0 | 50.0 | 56.0 | 83.0 | **93.0** |
| MNasNet | 11.0 | 70.0 | 65.0 | 50.0 | 50.0 | 59.0 | 91.0 |
| MobileNet | 74.0 | 69.0 | 76.0 | 56.0 | 62.0 | 75.0 | 73.0 |
| ResNet18 | 81.0 | 83.0 | 78.0 | 85.0 | 78.0 | 82.0 | 88.0 |
| SqeezeNet | 77.0 | 65.0 | 72.0 | 66.0 | 63.0 | 78.0 | 87.0 |
| ShuffleNet | 89.0 | 74.0 | 75.0 | 66.0 | 62.0 | 92.0 | **93.0** |

### APTOS Blindness
Performance comparison of the different architectures trained with the seven 
different processes (Base, PD: Plain Distillation, DD: Data Distillation, MD: Model 
Distillation, MDD: Model Data Distillation, FixMatch and MixMatch) in the APTOS Blindness dataset. 
In bold face the best model.

In this case we have a multiclass classification problem and then we have only used Accuracy
to compare the performance of the different architectures.

| Network | Base | PD | DD | MD | MDD | FixMatch | MixMatch |
|--|--|--|--|--|--|--|--|
| ResNet50 | 82.1 | - | - | - | - | - | - |
| ResNet101 | 83.0 | - | - | - | - | - | - |
| EfficientNet | 82.1 | - | - | - | - | - | - |
| FBNet | 83.0 | 83.0 | 83.3 | 83.3 | 83.7 | 83.6 | 78.5 |
| MixNet | 72.2 | 71.1 | 70.3 | 71.3 | 70.9 | 72.64 | 69.2 |
| MNasNet | 70.7 | 80.0 | 80.0 | 64.4 | 78.3 | 78.2 | 80.8 |
| MobileNet | 80.4 | 81.5 | 82.8 | 81.8 | 79.3 | **84.8** | 83.0 |
| ResNet18 | 83.3 | 82.5 | 82.1 | 82.5 | 82.9 | 84.7 | 70.5 |
| SqeezeNet | 78.8 | 82.1 | 79.5 | 81.0 | 81.0 | 82.2 | 73.3 |
| ShuffleNet | 72.9 | 72.8 | 72.8 | 73.9 | 72.4 | 72.6 | 65.5 |
