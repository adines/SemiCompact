from consistencyRegularisation.utils import *
from consistencyRegularisation.fixMatch import *
from consistencyRegularisation.mixMatch import *
from fastai.vision import *
from fastai.callbacks.tracker import *
import numpy as np



def fixMatch(targetModel, path, pathUnlabelled,outputPath, bs=32, size=224):
    if not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        labeled_data = (ImageList.from_folder(path)
                        .split_by_folder()
                        .label_from_folder()
                        .transform(get_transforms(), size=(size,size))
                        .databunch(bs=bs)
                        .normalize()
                        )

        unlabeled_data = get_unlabeled_data(pathUnlabelled, labeled_data, batch_multiplier=2,size=(size,size))
        model=getModel(targetModel,labeled_data.c)
        learn = Learner(labeled_data, model,
                        metrics=[accuracy]).fixmatch(unlabeled_data)
        save = SaveModelCallback(learn, monitor='accuracy', name='model-'+targetModel+'-fixmatch')
        early = EarlyStoppingCallback(learn, monitor='accuracy', patience=8)

        # learn.fine_tune(50,freeze_epochs=2)
        learn.fit_one_cycle(2, callbacks=[save, early])
        learn.unfreeze()
        learn.lr_find()
        lr = learn.recorder.lrs[np.argmin(learn.recorder.losses)]
        if lr < 1e-05:
            lr = 1e-03
        learn.fit_one_cycle(50, max_lr=lr / 10, callbacks=[save, early])
        learn.save(targetModel+'-fixmatch')
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        shutil.copy(path + os.sep + 'models' + os.sep + targetModel+'-fixmatch' + '.pth',
                    outputPath + os.sep + 'target_' + targetModel+'-fixmatch' + '.pth')


def mixMatch(targetModel, path, pathUnlabelled,outputPath, bs=32, size=224):
    if not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        labeled_data = (ImageList.from_folder(path)
                        .split_by_folder()
                        .label_from_folder()
                        .transform(get_transforms(), size=(size,size))
                        .databunch(bs=bs)
                        .normalize()
                        )
        unlabeled_data = (ImageList.from_folder(pathUnlabelled))
        model=getModel(targetModel,labeled_data.c)
        learn = Learner(labeled_data, model, metrics=[accuracy]).mixmatch(
            unlabeled_data, α=.75, λ=75)

        save = SaveModelCallback(learn, monitor='accuracy', name='model-'+targetModel+'-mixmatch')
        early = EarlyStoppingCallback(learn, monitor='accuracy', patience=8)

        # learn.fine_tune(50,freeze_epochs=2)
        learn.fit_one_cycle(2, callbacks=[save, early])
        learn.unfreeze()
        learn.lr_find()
        lr = learn.recorder.lrs[np.argmin(learn.recorder.losses)]
        if lr < 1e-05:
            lr = 1e-03
        learn.fit_one_cycle(50, max_lr=lr / 10, callbacks=[save, early])
        learn.save(targetModel + '-mixmatch')
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        shutil.copy(path + os.sep + 'models' + os.sep + targetModel + '-mixmatch' + '.pth',
                    outputPath + os.sep + 'target_' + targetModel + '-mixmatch' + '.pth')