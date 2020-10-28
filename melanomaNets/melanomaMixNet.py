from fastai.vision.all import *
import fastai
import torchvision.models as models
import sys
sys.path.insert(0, 'MixNet-PyTorch/')

from mixnet import MixNet

torch.cuda.set_device(2)

path="melanoma2"

bs=32
size=224

dls=ImageDataLoaders.from_folder(path,batch_tfms=aug_transforms(),item_tfms=Resize(size),bs=bs)

def create_squeezenet_body(cut=None):
  arch = "l"
  model = MixNet(arch=arch, num_classes=10).cuda()
  checkpoint = torch.load("MixNet-PyTorch/pretrained_weights/mixnet_{}_top1v_78.6.pkl".format(arch))
  pre_weight = checkpoint['model_state']
  model_dict = model.state_dict()
  pretrained_dict = {"module." + k: v for k, v in pre_weight.items() if "module." + k in model_dict}
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)

  model=nn.Sequential(*list(model.children())[:-1])

  return model


body = create_squeezenet_body()

nf = num_features_model(nn.Sequential(*body.children())) * (2)
head = create_head(nf, dls.c)
model = nn.Sequential(body, head)

apply_init(model[1], nn.init.kaiming_normal_)


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


#weights = get_weights(dls) 
#class_weights = torch.FloatTensor(weights).to(dls.device)

learn = Learner(dls, model,splitter=default_split, metrics=[accuracy,F1Score(),Precision(),Recall()])
#learn.loss_func = partial(F.cross_entropy, weight=class_weights)

learn.fine_tune(10)
learn.save('melanomaMixNet')
