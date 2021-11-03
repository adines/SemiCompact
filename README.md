# Semi-Supervised Learning for Image Classification using Compact Networks in the Medical Context
The development of mobile and on the edge applications that embed deep convolutional neural models has the potential to revolutionise healthcare. However, most deep learning models require computational resources that are not available in smartphones or edge devices; an issue that can be faced by means of compact models that require less resources than standard deep learning models. The problem with such models is that they are, at least usually, less accurate than bigger models. We address the accuracy limitation of compact networks with the application of semi-supervised learning techniques, which take advantage of unlabelled data. In particular, we study the application of self-training methods, consistency regularisation techniques and quantization techniques. In addition, we have developed a Python library in order to facilitate the combination of compact networks and semi-supervised learning methods to tackle image classification tasks. We present a thorough analysis for the results obtained by combining 9 compact networks and 6 semi-supervised processes when applied to 10 biomedical datasets. In particular, we first compare the performance of the networks when training using only labelled data, and, we observe that there are not significant differences between FBNet, MixNet, MNasNet and ResNet18 compact networks and standard size models. Then, we study the impact of applying the different semi-supervised methods, and we can conclude that combining Data Distillation and MixNet, and Plain Distillation and ResNet18 the best results are obtained. Finally, we analyse the efficiency of each network and we can conclude that compact networks outperforms standard size networks.


## Installation
In this api we present different semi-supervised methods grouped into two different families: self-training methods and consistency-regularization methods.

Self-training methods are available in PyPi for Python 3.6 and Fastai v2. To use it, you have to install Python 3.6 and pip.
````
    sudo pip3 install compact-distillation
````

Consistency-regularization methods are available in PyPi for Python 3.6 and Fastai v1. To use it, you have to install Python 3.6 and pip.
````
    sudo pip3 install compact-consistencyReg
````

## Semi-supervised Methods
In this api we present different semi-supervised methods grouped into two different families: self-training methods and consistency-regularization methods.

### Self-training methods
The self-training methods are the following, for more information of this methods see [this](https://www.sciencedirect.com/science/article/pii/S0169260720316151):
- Plain Distillation $\rightarrow$ plainDistillation(baseModel, targetModel, path, pathUnlabelled, outputPath,confidence)
- Data Distillation $\rightarrow$ dataDistillation(baseModel, targetModel, transforms, path, pathUnlabelled, outputPath, confidence)
- Model Distillation $\rightarrow$ modelDistillation(baseModels, targetModel, path, pathUnlabelled, outputPath, confidence)
- ModelData Distillation $\rightarrow$ modelDataDistillation(baseModels, targetModel,transforms, path, pathUnlabelled, outputPath, confidence)


### Consistency-regularization methods
The consistency regularization methods are the following, for more information see this [link](https://proceedings.neurips.cc/paper/2016/file/30ef30b64204a3088a26bc2e6ecf7602-Paper.pdf):
- FixMatch $\rightarrow$ FixMatch(targetModel, path, pathUnlabelled, outputPath)
- MixMatch $\rightarrow$ MixMatch(targetModel, path, pathUnlabelled, outputPath)


## Networks
In this work, we explore a variety of manually designed architectures, automatically designed architectures and quantized networks. 
Namely, we have employed 4 manually designed compact networks, 3 automatically desiged 
networks and 2 quantized networks. In addition, for our experiments, we have considered three standard size networks 
that are Resnet-50 and Resnet-101, and EfficientNet-B3. We provide a comparison of different 
features of these networks in the following table.

| Network | Params (M) | FLOPs (M) | Top-1 acc (%) | Top-5 acc (%) | Design | Implementation |
|--|--|--|--|--|--|--|
| ResNet50 | 26 | 4100 | 76.0 | 93.0 | Manual | [Official Pytorch](https://pytorch.org/vision/stable/models.html) |
| ResNet101 | 44 | 8540 | 80.9 | 95.6 | Manual | [Official Pytorch](https://pytorch.org/vision/stable/models.html) |
| EfficientNet | 12 | 1800 | 81.6 | 95.7 | Auto | [Unofficial Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) |
| FBNet | 9.4 | 753 | 78.9 | 94.3 | Auto | [Timm model](https://rwightman.github.io/pytorch-image-models/models/)
| MixNet | 5 | 360 | 78.9 | 94.2 | Auto | [Timm model](https://rwightman.github.io/pytorch-image-models/models/)
| MNasNet | 5.2 | 403 | 75.6 | 92.7 | Auto | [Timm model](https://rwightman.github.io/pytorch-image-models/models/) |
| MobileNet | 3.4 | 300 | 74.7 | 92.5 | Manual | [Official Pytorch](https://pytorch.org/vision/stable/models.html) |
| ResNet18 | 11 | 1300 | 69.6 | 89.2 | Manual | [Official Pytorch](https://pytorch.org/vision/stable/models.html)|
| SqeezeNet | 1.3 | 833 | 57.5 | 80.3 | Manual | [Official Pytorch](https://pytorch.org/vision/stable/models.html) |
| ShuffleNet | 5.3 | 524 | 69.4 | 88.3 | Manual | [Official Pytorch](https://pytorch.org/vision/stable/models.html) |
| ReNet18 Quantized | 11 | - | 69.5 | 88.9 | Manual | [Official Pytorch](https://pytorch.org/docs/stable/torchvision/models.html) |
| ResNet50 Quantized | 26 | - | 75.9 | 92.8 | Manual | [Official Pytorch](https://pytorch.org/docs/stable/torchvision/models.html) |


## Transforms
The transformations available in the API to apply in the data augmentation methods are:
- H Flip
- V Flip 
- H+V Flip
- Blurring
- Gamma
- Gaussian Blur
- Median Blur
- Bilateral Filter
- Equalize histogram
- 2D-Filter'


## Datasets
In this work, we propose a benchmark of 10 partially annotated biomedical datasets, described in the following Table, and evaluate the performance of deep learning models and semi-supervised methods using such a benchmark.

| Dataset | Number of Images | Number of  Classes | Description| Split | Reference |
|--|--|--|--|--|--|
| Blindness| 3662 | 5 | Diabetic retinopathy images| [Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EULlOvNF4ktMnxEx7l3QL04BIrAEBLXFgV80kc2wWBLj1Q?download=1) |[View](https://www.kaggle.com/c/aptos2019-blindness-detection)|
| Chest X Ray| 2355 | 2 | Chest X-Rays images| [Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EU5FdFSUqvRIoJNl2-YiYEMB9DP4_LZ0NNs2v6KP0WB5SA?download=1)| [View](http://www.sciencedirect.com/science/article/pii/S0092867418301545)|
| Fungi| 1204 | 4 | Dye decolourisation of fungal strain| [Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/ESoQP2IaXX1OpwhH8PINbUABsBmYlPK0ju7Pf5VLA37hZQ?download=1)|[View](https://link.springer.com/article/10.1007/s00500-019-03832-8)|
| HAM 10000 | 10015 | 7 | Dermatoscopic images of skin lesions|[Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EbTiHB68w-5OuayrdUAa-CwBKeTsfJg7Hmdf9gGXqIH-Ig?download=1)|[View](https://www.nature.com/articles/sdata2018161)|
| ISIC | 1500 | 7 | Colour images of skin lesions|[Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/ES74_yivyrZEukq5V6U_oWMBM2QkMSnPShKJKWiKy8fqgg?download=1)|[View](https://arxiv.org/pdf/1710.05006.pdf)|
| Kvasir  | 8000 | 8 | Gastrointestinal disease images|[Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EYUWxPBUecpLpCTn-LSOHysB5DhP2QPabRjX_BxwJRyuEg?download=1)|[View](https://dl.acm.org/doi/10.1145/3083187.3083212)|
| Open Sprayer| 6697 | 2 | Dron pictures of broad leaved docks|[Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/ESjVRz_J7aNPrO3_Py-J158B59R36h5ET42ifNiWRyLQAg?download=1)|[View](https://www.kaggle.com/gavinarmstrong/open-sprayer-images)|
| Plants | 5500 | 12 | Colour images of plants|[Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EYhFL8NnxRROqnBQMXkVnJUBKOkH_OkJdWPqG4KsgWZaxQ?download=1)|[View](https://arxiv.org/pdf/1711.05458.pdf)|
| Retinal OCT | 84484 | 4 | Retinal OCT images|[Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EVg_TNizZpFMnJkJQTBcuecBNzzULd7HxuHCNToInZU6gQ?download=1)|[View](http://www.sciencedirect.com/science/article/pii/S0092867418301545)|
| Tobacco  | 3492 | 10 | Document images|[Download](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EXdfQbAxx1lHn7PPBjWMotABZChHuV_BpkD0RTmZ94cM4Q?download=1)|[View](http://www.sciencedirect.com/science/article/pii/S0167865513004224)|

For our study, we have split each of the datasets of the benchmark into two different sets: a training set with the 75 % of images and a testing set with the 25 % of the images. In addition, for each dataset we have selected 75 images per class using them as labelled images and leaving the rest of the training images as unlabelled images to apply the semi-supervised learning methods.

## Results
In the following table we show the Mean (and standard deviation) F1-score for the different studied models for the base training method.

| Network | Blindness | Chest X Ray | Fungi | HAM 10000 | ISIC | Kvasir | Open Sprayer | Plants | Retinal OCT | Tobacco | Mean(std)||
|--|--|--|--|--|--|--|--|--|--|--|--|
|ResNet-50 | 59.3 | 89.9 | 91.0 | 54.3 | 87.6 | 89.0 | 91.3 | 84.3 | 97.4 | 81.8 | 82.5(13.5)|
|ResNet-101 | 58.2 | 90.7 | 86.9 | 52.0 | 84.0 | 83.8 | 95.8 | 84.3 | 96.4 | 80.1 | 81.2(14.1)|
|EfficientNet | 53.6 | 84.1 | 84.7 | 52.8 | 85.0 | 85.4 | 96.8 | 84.0 | 98.1 | 72.9 | 79.7(14.8)|
|--|--|--|--|--|--|--|--|--|--|--|--|
|FBNet | 57.5 | 87.4 | 89.0 | 47.2 | 85.2 | 88.9 | 95.4 | 81.8 | 94.9 | 73.3 | 80.1(15.3)|
|MixNet | 61.8 | 89.5 | 89.7 | 46.9 | 89.9 | 86.8 | 95.5 | 86.2 | 98.9 | 76.7 | 82.2(15.3)|
|MNasNet | 56.2 | 89.2 | 90.3 | 55.8 | 81.9 | 84.6 | 95.7 | 82.5 | 97.4 | 75.3  | 80.9(13.9)|
|MobileNet | 52 | 86.9 | 89.0 | 46.7 | 84.1 | 82.1 | 89.1 | 82.9 | 91.0 | 69.4 | 77.3(15.1)|
|ResNet-18 | 56.3 | 90.3 | 94.2 | 53.7 | 86.8 | 84.1 | 91.6 | 80.0 | 97.7 | 77.5 | 81.2(14.4)|
|SqueezeNet | 50.3 | 88.3 | 79.3 | 43.6 | 76.8 | 80.1 | 90.9 | 78.9 | 93.2 | 75.5 | 75.7(15.5)|
|ShuffleNet | 39.5 | 85.7 | 69.9 | 37.6 | 78.9 | 67.0 | 89.6 | 51.9 | 33.9 | 40.7 | 59.5(20.2)|
|--|--|--|--|--|--|--|--|--|--|--|--|
|ResNet-18 quantized | 45.1 | 77.8 | 88.1 | 47.0 | 86.5 | 84.2 | 91.3 | 75.1 | 91.6 | 55.8 | 74.3(17.2)| 
|ResNet-50 quantized | 48.6 | 77.2 | 83.2 | 42.9 | 78.6 | 81.1 | 85.4 | 77.7 | 91.6 | 69.7 | 73.6(15.0)|

Full results for each dataset can be found in the results folder.

## Examples of use
We have created two notebooks with a small example of how to build and execute the API methods.
- [Sel-Training example](https://colab.research.google.com/drive/1p_97Kvwmb_gMIoBfn2T_Zc204FYREc4A?usp=sharing)
- [Consistency-Regularisation example](https://colab.research.google.com/drive/1RH_CKva0kOMcJyYUGuj6AGBKht-9RwTR?usp=sharing)


