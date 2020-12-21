import cv2
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from fastai.vision import *
from fastai.callbacks.hooks import *
import numpy as np
import torch
from fastai.vision.learner import has_pool_type



availableModels=['ResNet18','ResNet50','ResNet101','EfficientNet','FBNet','MixNet','MNasNet','MobileNet','SqueezeNet','ShuffleNet']
availableTransforms=['H Flip','V Flip','H+V Flip','Blurring','Gamma','Gaussian Blur','Median Blur','Bilateral Filter','Equalize histogram','2D-Filter']

def testNameModel(model):
    return model in availableModels

def testPath(path):
    if os.path.isdir(path+os.sep+'train') and os.path.isdir(path+os.sep+'valid'):
        if len(os.listdir(path+os.sep+'train'))>0 and len(os.listdir(path+os.sep+'valid'))>0 and len(os.listdir(path+os.sep+'train'))==len(os.listdir(path+os.sep+'valid')):
            return True
        else:
            return False
    else:
        return False

def testTransforms(transforms):
    for transform in transforms:
        if not transform in availableTransforms:
            return False
    return True


# Modelos
def create_fbnet(num_classes):
    if not os.path.exists('mobile_vision'):
        os.system("git clone https://github.com/facebookresearch/mobile-vision.git")
        sys.path.insert(0, 'mobile_vision/')
    from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
    model = fbnet("dmasking_l3", pretrained=True)
    body=model.backbone
    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    head = create_head(nf, num_classes)
    model = nn.Sequential(body, head)
    apply_init(model[1], nn.init.kaiming_normal_)
    return model

def create_resnet18(num_classes):
    return create_cnn_model(models.resnet18,num_classes,-2,ps=0.5)

def create_resnet50(num_classes):
    return create_cnn_model(models.resnet50,num_classes,-2,ps=0.5)

def create_resnet101(num_classes):
    return create_cnn_model(models.resnet101,num_classes,-2,ps=0.5)

def create_efficientnet(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b3',num_classes=num_classes)

def create_mixnet(num_classes):
    if not os.path.exists('MixNet-PyTorch'):
        os.system("git clone https://github.com/ansleliu/MixNet-PyTorch.git")
        sys.path.insert(0, 'MixNet-PyTorch/')
    from mixnet import MixNet
    arch = "l"
    body = MixNet(arch=arch, num_classes=num_classes).cuda()
    checkpoint = torch.load("MixNet-PyTorch/pretrained_weights/mixnet_{}_top1v_78.6.pkl".format(arch))
    pre_weight = checkpoint['model_state']
    model_dict = body.state_dict()
    pretrained_dict = {"module." + k: v for k, v in pre_weight.items() if "module." + k in model_dict}
    model_dict.update(pretrained_dict)
    body.load_state_dict(model_dict)
    body = nn.Sequential(*list(body.children())[:-1])
    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    head = create_head(nf, num_classes)
    model = nn.Sequential(body, head)
    apply_init(model[1], nn.init.kaiming_normal_)
    return model


def create_mnasnet(num_classes):
    body = models.mnasnet1_0(pretrained=True).layers
    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    head = create_head(nf, num_classes)
    model = nn.Sequential(body, head)
    apply_init(model[1], nn.init.kaiming_normal_)
    return model

def create_mobilenet(num_classes):
    body = models.mobilenet_v2(pretrained=True).features

    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    head = create_head(nf, num_classes)
    model = nn.Sequential(body, head)

    apply_init(model[1], nn.init.kaiming_normal_)
    return model

def create_shufflenet(num_classes):
    cut=None
    body = models.shufflenet_v2_x1_0(pretrained=True)
    if cut is None:
        ll = list(enumerate(body.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        body=nn.Sequential(*list(body.children())[:cut])
    elif callable(cut):
        body= cut(body)
    else:
        raise Exception("cut must be either integer or function")

    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    head = create_head(nf, num_classes)
    model = nn.Sequential(body, head)

    apply_init(model[1], nn.init.kaiming_normal_)
    return model

def create_squeezenet(num_classes):
    cut=None
    body = models.squeezenet1_0(pretrained=True)
    if cut is None:
        ll = list(enumerate(body.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        body= nn.Sequential(*list(body.children())[:cut])
    elif callable(cut):
        body= cut(body)
    else:
        raise Exception("cut must be either integer or function")
    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    head = create_head(nf, num_classes)
    model = nn.Sequential(body, head)

    apply_init(model[1], nn.init.kaiming_normal_)
    return model


def getModel(model,numClasses):
    modelo='create_'+model.lower()
    method=getattr(importlib.import_module("consistencyRegularisation.utils"),modelo)
    return method(numClasses)



def getTransform(transform, image):
    if transform=="H Flip":
        return cv2.flip(image,0)
    elif transform=="V Flip":
        return cv2.flip(image,1)
    elif transform=="H+V Flip":
        return cv2.flip(image,-1)
    elif transform=="Blurring":
        return cv2.blur(image,(5,5))
    elif transform=="Gamma":
        invGamma = 1.0
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        return cv2.LUT(image, table)
    elif transform=="Gaussian Blur":
        return cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
    elif transform=="Median Blur":
        return cv2.medianBlur(image,5)
    elif transform=="Bilateral Filter":
        return cv2.bilateralFilter(image,9,75,75)
    elif transform=="Equalize histogram":
        equ_im = cv2.equalizeHist(image)
        return np.hstack((image, equ_im))
    elif transform=="2D-Filter":
        kernel = np.ones((5, 5), np.float32) / 25
        return cv2.filter2D(image, -1, kernel)