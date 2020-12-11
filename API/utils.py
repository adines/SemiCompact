import os
import glob
import shutil
import numpy as np
import cv2

# Imports adicionales
from fastai.vision.all import *
from fastai.vision.learner import *
import fastai
import torchvision.models as models
import sys
sys.path.insert(0, '../melanoma/mobile_vision/')

from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from efficientnet_pytorch import EfficientNet

sys.path.insert(0, '../melanoma/MixNet-PyTorch/')

from mixnet import MixNet
import importlib




availableModels=['ResNet18','ResNet50','ResNet101','EfficientNet','FBNet','MixNet','MNasNet','MobileNet','SqueezeNet','ShuffleNet']


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


# Distillation
def moda(lista):
    tam=len(lista[0][2])
    x=np.zeros(tam)
    for l in lista:
        x=x+l[2].numpy()
    x=x/len(lista)
    maximo=x.argmax()
    return maximo, x[maximo]


def omniModel(path, pathUnlabelled,learners,th):
    images=sorted(glob.glob(pathUnlabelled+os.sep+"*"))
    i=path.rfind(os.sep)
    newPath=path[:i] + "_tmp"
    shutil.copytree(path, newPath)
    for image in images:
        lista=[]
        for learn in learners:
            p=learn.predict(image)
            lista.append(p)
        mod, predMax=moda(lista)
        if predMax>th:
            shutil.copyfile(image,newPath+os.sep+'train'+os.sep+learn.dls.vocab[mod]+os.sep+learn.dls.vocab[mod]+'_'+image.split(os.sep)[-1])
            print(image+" --> "+newPath+os.sep+'train'+os.sep+learn.dls.vocab[mod]+os.sep+learn.dls.vocab[mod]+'_'+image.split(os.sep)[-1])

def omniData(path, pathUnlabelled,learn, transforms,th):
    images=sorted(glob.glob(pathUnlabelled+os.sep+"*"))
    i = path.rfind(os.sep)
    newPath = path[:i] + "_tmp"
    shutil.copytree(path, newPath)

    for image in images:

    # Cambiar por las transformaciones pasada
        im=cv2.imread(image,1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        lista=[]
        n=im
        pn=learn.predict(n)
        lista.append(pn)

        h_im=cv2.flip(im,0)
        h=h_im
        ph=learn.predict(h)
        lista.append(ph)

        v_im=cv2.flip(im,1)
        pv=learn.predict(v_im)
        lista.append(pv)
        b_im=cv2.flip(im,-1)

        pb=learn.predict(b_im)
        lista.append(pb)
        blur_im=cv2.blur(im,(5,5))

        pblur=learn.predict(blur_im)
        lista.append(pblur)
        invGamma=1.0
        table=np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype('uint8')
        gamma_im = cv2.LUT(im, table)

        pgamma = learn.predict(gamma_im)
        lista.append(pgamma)
        gblur_im = cv2.GaussianBlur(im, (5, 5), cv2.BORDER_DEFAULT)

        pgblur = learn.predict(gblur_im)
        lista.append(pgblur)

        mod, predMax = moda(lista)
        if predMax > th:
            shutil.copyfile(image, newPath + os.sep + 'train' + os.sep + learn.dls.vocab[mod] + os.sep + learn.dls.vocab[
                mod] + '_' + image.split(os.sep)[-1])
            print(image + " --> " + newPath + os.sep + 'train' + os.sep + learn.dls.vocab[mod] + os.sep + learn.dls.vocab[
                mod] + '_' + image.split(os.sep)[-1])


def omniModelData(path, pathUnlabelled,learners,transforms,th):
    images = sorted(glob.glob(pathUnlabelled + os.sep + "*"))

    i = path.rfind(os.sep)
    newPath = path[:i] + "_tmp"
    shutil.copytree(path, newPath)
    for image in images:
        lista=[]
        for learn in learners:
            im=cv2.imread(image,1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          # n=Image(pil2tensor(im, dtype=np.float32).div_(255))
            pn=learn.predict(im)
            lista.append(pn)
            h_im=cv2.flip(im,0)
          #h=Image(pil2tensor(h_im, dtype=np.float32).div_(255))
            ph=learn.predict(h_im)
            lista.append(ph)
            v_im=cv2.flip(im,1)
          #v=Image(pil2tensor(v_im, dtype=np.float32).div_(255))
            pv=learn.predict(v_im)
            lista.append(pv)
            b_im=cv2.flip(im,-1)
          #b=Image(pil2tensor(b_im, dtype=np.float32).div_(255))
            pb=learn.predict(b_im)
            lista.append(pb)
            blur_im=cv2.blur(im,(5,5))
          #blur=Image(pil2tensor(blur_im, dtype=np.float32).div_(255))
            pblur=learn.predict(blur_im)
            lista.append(pblur)
            invGamma=1.0
            table=np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype('uint8')
            gamma_im=cv2.LUT(im,table)
          #gamma=Image(pil2tensor(gamma_im, dtype=np.float32).div_(255))
            pgamma=learn.predict(gamma_im)
            lista.append(pgamma)
            gblur_im=cv2.GaussianBlur(im,(5,5),cv2.BORDER_DEFAULT)
            #gblur=Image(pil2tensor(gblur_im, dtype=np.float32).div_(255))
            pgblur=learn.predict(gblur_im)
            lista.append(pgblur)
        mod, predMax=moda(lista)
        if predMax > th:
            shutil.copyfile(image, newPath + os.sep + 'train' + os.sep + learn.dls.vocab[mod] + os.sep + learn.dls.vocab[
                mod] + '_' + image.split(os.sep)[-1])
            print(image + " --> " + newPath + os.sep + 'train' + os.sep + learn.dls.vocab[mod] + os.sep + learn.dls.vocab[
                mod] + '_' + image.split(os.sep)[-1])

def plainSupervised(path,pathUnlabelled,learn,th):
    images=sorted(glob.glob(pathUnlabelled+os.sep+"*"))
    i = path.rfind(os.sep)
    newPath = path[:i] + "_tmp"
    shutil.copytree(path, newPath)
    for image in images:
        pn=learn.predict(image)
        if pn[2]>th:
            shutil.copyfile(image,newPath+os.sep+'train'+os.sep+pn[0]+os.sep+pn[0]+'_'+image.split(os.sep)[-1])
            print(image+" --> "+newPath+os.sep+'train'+os.sep+pn[0]+os.sep+pn[0]+'_'+image.split(os.sep)[-1])



# Modelos
def create_fbnet(num_classes):
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
    arch = "l"
    body = MixNet(arch=arch, num_classes=num_classes).cuda()
    checkpoint = torch.load("../melanoma/MixNet-PyTorch/pretrained_weights/mixnet_{}_top1v_78.6.pkl".format(arch))
    pre_weight = checkpoint['model_state']
    model_dict = body.state_dict()
    pretrained_dict = {"module." + k: v for k, v in pre_weight.items() if "module." + k in model_dict}
    model_dict.update(pretrained_dict)
    body.load_state_dict(model_dict)
    body = nn.Sequential(*list(body.children())[:-1])
    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    head = create_head(nf, dls.c)
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
        raise NamedError("cut must be either integer or function")

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
        raise NamedError("cut must be either integer or function")
    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    head = create_head(nf, num_classes)
    model = nn.Sequential(body, head)

    apply_init(model[1], nn.init.kaiming_normal_)
    return model


def getModel(model,numClasses):
    modelo='create_'+model.lower()
    method=getattr(importlib.import_module("API.utils"),modelo)
    return method(numClasses)