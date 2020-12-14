from API.utils import *
from fastai.vision.all import *
import fastai
import torchvision.models as models
from API.fixMatch import *


def distillation(baseModel, targetModel, path, pathUnlabelled, outputPath, bs=32, size=224, confidence=0.8):
    if not testNameModel(baseModel):
        print("The base model selected is not valid")
    elif not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        # Load images
        dls = ImageDataLoaders.from_folder(path, batch_tfms=aug_transforms(), item_tfms=Resize(size), bs=bs)

        # Load base model
        bModel=getModel(baseModel,dls.c)

        # Create base learner
        save = SaveModelCallback(monitor='accuracy', fname='model-'+baseModel)
        early = EarlyStoppingCallback(monitor='accuracy', patience=8)
        learn = Learner(dls, bModel, splitter=default_split, metrics=[accuracy], cbs=[early, save])

        # Train base learner
        learn.fine_tune(50, freeze_epochs=2)
        learn.save(outputPath+os.sep+baseModel)

        # supervised method
        plainSupervised(path,pathUnlabelled,learn,confidence)

        # Load new images
        dls2 = ImageDataLoaders.from_folder(path+'_tmp', batch_tfms=aug_transforms(), item_tfms=Resize(size), bs=bs)

        # Load base model
        tModel = getModel(targetModel, dls2.c)

        # Create base learner
        save2 = SaveModelCallback(monitor='accuracy', fname='model-' + targetModel)
        early2 = EarlyStoppingCallback(monitor='accuracy', patience=8)
        learn2 = Learner(dls2, tModel, splitter=default_split, metrics=[accuracy], cbs=[early2, save2])

        # Train base learner
        learn2.fine_tune(50, freeze_epochs=2)
        learn2.save(outputPath+os.sep+targetModel)
        shutil.rmtree(path+'_tmp')



def dataDistillation(baseModel, targetModel, transforms, path, pathUnlabelled, outputPath, bs=32, size=224, confidence=0.8):
    if not testNameModel(baseModel):
        print("The base model selected is not valid")
    elif not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    elif not testTransforms(transforms):
        print("There are invalid transforms")
    else:
        # Load images
        dls = ImageDataLoaders.from_folder(path, batch_tfms=aug_transforms(), item_tfms=Resize(size), bs=bs)

        # Load base model
        bModel = getModel(baseModel, dls.c)

        # Create base learner
        save = SaveModelCallback(monitor='accuracy', fname='model-' + baseModel)
        early = EarlyStoppingCallback(monitor='accuracy', patience=8)
        learn = Learner(dls, bModel, splitter=default_split, metrics=[accuracy], cbs=[early, save])

        # Train base learner
        learn.fine_tune(50, freeze_epochs=2)
        learn.save(outputPath + os.sep + baseModel)

        # supervised method
        omniData(path, pathUnlabelled, learn, transforms,confidence)

        # Load new images
        dls2 = ImageDataLoaders.from_folder(path + '_tmp', batch_tfms=aug_transforms(), item_tfms=Resize(size), bs=bs)

        # Load base model
        tModel = getModel(targetModel, dls2.c)

        # Create base learner
        save2 = SaveModelCallback(monitor='accuracy', fname='model-' + targetModel)
        early2 = EarlyStoppingCallback(monitor='accuracy', patience=8)
        learn2 = Learner(dls2, tModel, splitter=default_split, metrics=[accuracy], cbs=[early2, save2])

        # Train base learner
        learn2.fine_tune(50, freeze_epochs=2)
        learn2.save(outputPath + os.sep + targetModel)
        shutil.rmtree(path + '_tmp')


def modelDistillation(baseModels, targetModel, path, pathUnlabelled, outputPath, bs=32, size=224, confidence=0.8):
    for baseModel in baseModels:
        if not testNameModel(baseModel):
            print("The base model selected is not valid")
            return
    if not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        # Load images
        dls = ImageDataLoaders.from_folder(path, batch_tfms=aug_transforms(), item_tfms=Resize(size), bs=bs)

        # Load base model
        learners=[]
        for baseModel in baseModels:
            bModel = getModel(baseModel, dls.c)

            # Create base learner
            save = SaveModelCallback(monitor='accuracy', fname='model-' + baseModel)
            early = EarlyStoppingCallback(monitor='accuracy', patience=8)

            learn = Learner(dls, bModel, splitter=default_split, metrics=[accuracy], cbs=[early, save])

            # Train base learner
            learn.fine_tune(50, freeze_epochs=2)
            learn.save(outputPath + os.sep + baseModel)
            learners.append(learn)

        # supervised method
        omniModel(path, pathUnlabelled, learners, confidence)

        # Load new images
        dls2 = ImageDataLoaders.from_folder(path + '_tmp', batch_tfms=aug_transforms(), item_tfms=Resize(size), bs=bs)

        # Load base model
        tModel = getModel(targetModel, dls2.c)

        # Create base learner
        save2 = SaveModelCallback(monitor='accuracy', fname='model-' + targetModel)
        early2 = EarlyStoppingCallback(monitor='accuracy', patience=8)
        learn2 = Learner(dls2, tModel, splitter=default_split, metrics=[accuracy], cbs=[early2, save2])

        # Train base learner
        learn2.fine_tune(50, freeze_epochs=2)
        learn2.save(outputPath + os.sep + targetModel)
        shutil.rmtree(path + '_tmp')


def modelDataDistillation(baseModels, targetModel, transforms, path, pathUnlabelled, outputPath, bs=32, size=224, confidence=0.8):
    for baseModel in baseModels:
        if not testNameModel(baseModel):
            print("The base model selected is not valid")
            return
    if not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    elif not testTransforms(transforms):
        print("There are invalid transforms")
    else:
        # Load images
        learners=[]
        for baseModel in baseModels:
            dls = ImageDataLoaders.from_folder(path, batch_tfms=aug_transforms(), item_tfms=Resize(size), bs=bs)

            # Load base model
            bModel = getModel(baseModel, dls.c)

            # Create base learner
            save = SaveModelCallback(monitor='accuracy', fname='model-' + baseModel)
            early = EarlyStoppingCallback(monitor='accuracy', patience=8)
            learn = Learner(dls, bModel, splitter=default_split, metrics=[accuracy], cbs=[early, save])

            # Train base learner
            learn.fine_tune(50, freeze_epochs=2)
            learn.save(outputPath + os.sep + baseModel)
            learners.append(learn)

        # supervised method
        omniModelData(path, pathUnlabelled, learners, transforms,confidence)

        # Load new images
        dls2 = ImageDataLoaders.from_folder(path + '_tmp', batch_tfms=aug_transforms(), item_tfms=Resize(size), bs=bs)

        # Load base model
        tModel = getModel(targetModel, dls2.c)

        # Create base learner
        save2 = SaveModelCallback(monitor='accuracy', fname='model-' + targetModel)
        early2 = EarlyStoppingCallback(monitor='accuracy', patience=8)
        learn2 = Learner(dls2, tModel, splitter=default_split, metrics=[accuracy], cbs=[early2, save2])

        # Train base learner
        learn2.fine_tune(50, freeze_epochs=2)
        learn2.save(outputPath + os.sep + targetModel)
        shutil.rmtree(path + '_tmp')



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
        save = SaveModelCallback(learn, monitor='accuracy', name=outputPath+os.sep+'model-'+targetModel+'-fixmatch')
        early = EarlyStoppingCallback(learn, monitor='accuracy', patience=8)

        # learn.fine_tune(50,freeze_epochs=2)
        learn.fit_one_cycle(2, callbacks=[save, early])
        learn.unfreeze()
        learn.lr_find()
        lr = learn.recorder.lrs[np.argmin(learn.recorder.losses)]
        if lr < 1e-05:
            lr = 1e-03
        learn.fit_one_cycle(50, max_lr=lr / 10, callbacks=[save, early])
        learn.save(outputPath+os.sep+targetModel+'-fixmatch')


def mixMatch(targetModel, path, pathUnlabelled,outputPath, bs=32, size=224):
    if not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        pass