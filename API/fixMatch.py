from functools import partial

import fastai
from fastai.vision import *
from fastai.callbacks.tracker import *
from fastai.callbacks.hooks import *

from fastai.basic_train import LearnerCallback, Learner, DataBunch, SmoothenValue, to_data, functools, \
    add_metrics, Module, nn
from fastai.imports import torch, F

import numpy as np
from torch.utils.data import Dataset

from numbers import Integral
import gc

import torchvision
import torch


class MultiTfmLabelList(LabelList):
    def __init__(self, x:ItemList, y:ItemList, tfms:TfmList=None, tfm_y:bool=False, K=2, **kwargs):
        "K: number of transformed samples generated per item"
        self.x,self.y,self.tfm_y,self.K = x,y,tfm_y,K
        self.y.x = x
        self.item=None
        self.transform(tfms, **kwargs)

    def __getitem__(self,idxs:Union[int, np.ndarray])->'LabelList':
        "return a single (x, y) if `idxs` is an integer or a new `LabelList` object if `idxs` is a range."
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None: x,y = self.x[idxs],self.y[idxs]
            else:                 x,y = self.item   ,0
            if self.tfms or self.tfmargs:
                x = [x.apply_tfms(self.tfms, **self.tfmargs) for _ in range(self.K)]
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            if y is None: y=0
            return x,y
        else: return self.new(self.x[idxs], self.y[idxs])

def MultiCollate(batch):
    batch = to_data(batch)
    if isinstance(batch[0][0],list): batch = [[torch.stack(s[0]),s[1]] for s in batch]
    return torch.utils.data.dataloader.default_collate(batch)


class FixMatchLoss(Module):

    def __init__(self, reduction='mean', unlabeled_loss_coeff=1.0, threshold=0.95):
        super().__init__()
        crit = nn.CrossEntropyLoss()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else:
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction
        self.unlabeled_loss_coeff = unlabeled_loss_coeff
        self.threshold = threshold

    def forward(self, preds, target, bs=None):

        if bs is None: return F.cross_entropy(preds, target)

        # labeled_preds = torch.log_softmax(preds[:bs], dim=1)
        # Lx = -(labeled_preds * target[:bs]).sum(dim=1).mean()
        # Lx = -(labeled_preds[range(labeled_preds.shape[0]), target[:bs]]).mean()

        Lx = F.cross_entropy(preds[:bs], target[:bs])
        self.Lx = Lx.item()

        logits_u_w, logits_u_s = preds[bs:].chunk(2)

        pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

        self.Lu = (Lu * self.unlabeled_loss_coeff).item()

        return Lx + Lu * self.unlabeled_loss_coeff

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit


class FixMatchCallback(LearnerCallback):
    _order = -20

    def __init__(self,
                 learn: Learner,
                 unlabeled_data: DataBunch,
                 unlabeled_loss_coeff: float = 1):
        super().__init__(learn)

        self.learn, self.unlabeled_loss_coeff = learn, unlabeled_loss_coeff
        self.unlabeled_dl = unlabeled_data.train_dl
        self.n_classes = unlabeled_data.c
        self.unlabeled_data = unlabeled_data

    def on_train_begin(self, n_epochs, **kwargs):
        self.learn.loss_func = FixMatchLoss(unlabeled_loss_coeff=self.unlabeled_loss_coeff)
        self.uldliter = iter(self.unlabeled_dl)
        self.smoothLx, self.smoothLu = SmoothenValue(0.98), SmoothenValue(0.98)
        self.recorder.add_metric_names(["train_Lx", "train_Lu*λ"])
        self.it = 0
        print('labeled dataset     : {:13,} samples'.format(len(self.learn.data.train_ds)))
        print('unlabeled dataset   : {:13,} samples'.format(len(self.unlabeled_data.train_ds)))
        print("labeled batch size:", learn.data.batch_size)
        print("unlabeled batch size:", unlabeled_data.batch_size)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        # Augmentation should already be applied in dataloader
        if not train: return

        ## UNLABELED
        try:
            # (inputs_u_w, inputs_u_s), _ = next(self.uldliter)
            # (batch_size, n_img, channel, height x width)

            img_pairs, _labels = next(self.uldliter)
            weak_imgs, strong_imgs = torch.split(img_pairs, 1, dim=1)
            inputs_u_w = weak_imgs.squeeze()
            inputs_u_s = strong_imgs.squeeze()

            gc.collect()
            torch.cuda.empty_cache()

        except StopIteration as exc:
            self.uldliter = iter(self.unlabeled_dl)

            # (inputs_u_w, inputs_u_s), _ = next(self.uldliter)

            img_pairs, _labels = next(self.uldliter)
            weak_imgs, strong_imgs = torch.split(img_pairs, 1, dim=1)
            inputs_u_w = weak_imgs.squeeze()
            inputs_u_s = strong_imgs.squeeze()

        bs = len(last_input)

        # LABELED
        inputs = torch.cat((last_input, inputs_u_w, inputs_u_s))

        gc.collect()
        torch.cuda.empty_cache()

        return {"last_input": inputs, "last_target": (last_target, bs)}

    def on_batch_end(self, train, **kwargs):
        if not train: return
        self.smoothLx.add_value(self.learn.loss_func.Lx)
        self.smoothLu.add_value(self.learn.loss_func.Lu)
        self.it += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, [self.smoothLx.smooth, self.smoothLu.smooth])

    def on_train_end(self, **kwargs):
        """At the end of training, loss_func and data are returned to their original values,
        and this calleback is removed"""
        self.learn.loss_func = self.learn.loss_func.get_old()
        drop_cb_fn(self.learn, 'FixMatchCallback')

def drop_cb_fn(learn, cb_name: str) -> None:
    cbs = []
    for cb in learn.callback_fns:
        if isinstance(cb, functools.partial):
            cbn = cb.func.__name__
        else:
            cbn = cb.__name__
        if cbn != cb_name: cbs.append(cb)
    learn.callback_fns = cbs


def fixmatch(learn: Learner, u_databunch: DataBunch, num_workers: int = None, unlabeled_loss_coeff: float = 1) -> Learner:
    labeled_data = learn.data
    learn.unlabeled_data = u_databunch
    if num_workers is None: num_workers = 1
    labeled_data.train_dl.num_workers = num_workers
    bs = labeled_data.train_dl.batch_size
    learn.callback_fns.append(partial(FixMatchCallback, unlabeled_data=u_databunch, unlabeled_loss_coeff=unlabeled_loss_coeff))
    return learn

Learner.fixmatch = fixmatch

class MultiTfmPairLabelList(LabelList):
    def __init__(self, x:ItemList, y:ItemList,
                 weak_tfms:TfmList=None, strong_tfms:TfmList=None, extra_weak_tfms=None, extra_strong_tfms=None, tfm_y:bool=False,
                 K=2, **kwargs):
        "K: number of transformed samples generated per item"
        self.x,self.y,self.tfm_y,self.K = x,y,tfm_y,K
        self.y.x = x
        self.item=None
        # self.transform(tfms, **kwargs)
        self.weak_tfms, self.strong_tfms = weak_tfms, strong_tfms
        self.extra_weak_tfms, self.extra_strong_tfms = extra_weak_tfms, extra_strong_tfms

    def __getitem__(self,idxs:Union[int, np.ndarray])->'LabelList':
        "return a single (x, y) if `idxs` is an integer or a new `LabelList` object if `idxs` is a range."
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None: x,y = self.x[idxs],self.y[idxs]
            else:                 x,y = self.item   ,0
            if self.weak_tfms and self.strong_tfms:
                # x = [x.apply_tfms(self.tfms, **self.tfmargs) for _ in range(self.K)]
                # x = [x.apply_tfms(self.weak_tfms), x.apply_tfms(self.strong_tfms)]
                x = (
                      x.apply_tfms(self.weak_tfms, size=224, resize_method=ResizeMethod.SQUISH),
                      x.apply_tfms(self.strong_tfms, size=224, resize_method=ResizeMethod.SQUISH))
            if self.extra_weak_tfms:
                x = (self.extra_weak_tfms(x[0].data), x[1])
            if self.extra_strong_tfms:
                x = (x[0], self.extra_strong_tfms(x[1].data))

            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            gc.collect()
            torch.cuda.empty_cache()
            if y is None: y=0
            return x,y
        else: return self.new(self.x[idxs], self.y[idxs])


def get_weak_transforms():
  weak_transforms = get_transforms()[0]
  return weak_transforms

def get_strong_transforms():
  strong_transforms = get_transforms(max_rotate=20, max_zoom=1.2,
    max_lighting=0.5, max_warp=0.5,
  )[0]
  return strong_transforms

def get_extra_strong_transforms():
  custom_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomCrop(300, pad_if_needed=True, padding_mode="reflect"),
    torchvision.transforms.Resize(train_image_size),
    torchvision.transforms.ColorJitter(0, 0, 0.9, 0.2),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.ToPILImage(),
  ])
  return custom_transforms


def get_unlabeled_data(unlabeled_dir, labeled_data, n_augment=2, batch_multiplier=2):
    u_image_list = ImageList.from_folder(unlabeled_dir).split_none()
    u_image_list.train._label_list = partial(MultiTfmPairLabelList, K=n_augment,
                                             weak_tfms=get_weak_transforms(), strong_tfms=get_strong_transforms(),
                                             extra_strong_tfms=get_extra_strong_transforms())
    u_databunch = (u_image_list.label_empty()
                   .databunch(
        bs=labeled_data.batch_size * batch_multiplier,
        collate_fn=MultiCollate)
                   .normalize(labeled_data.stats))
    u_databunch.c = 1
    return u_databunch


unlabeled_data = get_unlabeled_data('datasetRetina/', labeled_data, batch_multiplier=2)
