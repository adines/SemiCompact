from fastai.vision.all import *
import fastai
import torchvision.models as models


torch.cuda.set_device(2)

path="melanoma2"

bs=32
size=224

dls=ImageDataLoaders.from_folder(path,batch_tfms=aug_transforms(),item_tfms=Resize(size),bs=bs)

def create_squeezenet_body(cut=None):
  model = models.shufflenet_v2_x1_0(pretrained=True)
  if cut is None:
    ll = list(enumerate(model.children()))
    cut = next(i for i,o in reversed(ll) if has_pool_type(o))
  if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
  elif callable(cut): return cut(model)
  else: raise NamedError("cut must be either integer or function")

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
learn.save('melanomaShufflenet')
