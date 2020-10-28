from fastai.vision.all import *
import fastai

torch.cuda.set_device(2)

path="melanoma2"

bs=32
size=224

data=ImageDataLoaders.from_folder(path,batch_tfms=aug_transforms(),item_tfms=Resize(size),bs=bs)

def get_weights(dls):
    classes = [0,1]
    print(classes)
    #Get label ids from the dataset using map
    #train_lb_ids = L(map(lambda x: x[1], dls.train_ds))
    # Get the actual labels from the label_ids & the vocab
    #train_lbls = L(map(lambda x: classes[x], train_lb_ids))

    #Combine the above into a single
    train_lbls = L(map(lambda x: classes[x[1]], dls.train_ds))
    label_counter = Counter(train_lbls)
    n_most_common_class = max(label_counter.values()); 
    print(f'Occurrences of the most common class {n_most_common_class}')
    
    weights = [n_most_common_class/v for k, v in label_counter.items() if v > 0]
    return weights 


#weights = get_weights(data) 
#class_weights = torch.FloatTensor(weights).to(data.device)

learn=cnn_learner(data,resnet50,metrics=[accuracy,F1Score(),Precision(),Recall()])
#learn.loss_func = partial(F.cross_entropy, weight=class_weights)

learn.fine_tune(10)
