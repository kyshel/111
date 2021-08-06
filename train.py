# %% preset

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from shutil import copyfile
import random  
from torchsummary import summary
from collections import Counter
import yaml
import os
import pandas as pd
import numpy as np
import argparse
import json
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import copy 
from copy import deepcopy
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim import lr_scheduler
import util
import ax
from importlib import reload
# reload(util)
# reload(ax)
# reload(models) # affect repro maybe 
import  itertools 
import sys
import logging
from util import increment_path,strip_optimizer,colorstr
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# logger 
logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            # logging.FileHandler(save_dir / 'logger.txt'), # should get dir after 
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% functions

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def plot_cls_bar(cls_list, save_dir, dataset = None):
    # plot bar of classes
    data =  Counter(cls_list)
    cls_id_list = list(data.keys())
    num_list = list(data.values())
    x = cls_id_list
    y = num_list
    if dataset:
        if hasattr(dataset, 'cls_names'):
            x = [str(id) + '' + dataset.cls_names[id]  for id in cls_id_list]
 
    # https://stackoverflow.com/questions/19576317/matplotlib-savefig-does-not-save-axes/67369132#67369132
    fig = plt.figure(facecolor=(1, 1, 1) )
    # creating the bar plot
    plt.bar(x, y, color ='maroon',
            width = 0.4)
    plt.xlabel("Classes")
    plt.ylabel("No. of class")
    plt.title("Class distribute")
    
    save_fp = os.path.join(str(save_dir), "classes_distribute.png")
    plt.savefig(save_fp, bbox_inches='tight')
    plt.close(fig)
    # plt.show()



def isinteractive():  # unreliable!
    # Warning! this may determine wrong
    # pylint: disable=E0602
    # pylint: disable=undefined-variable
    # pylint: disable=reportUndefinedVariable
    
    
  

    try:
        # pylint: disable=E0602
        # pylint: disable=undefined-variable
        # pylint: disable=reportUndefinedVariable
        # get_ipython = globals()["get_ipython"]()
        get_ipython = vars(__builtins__)['get_ipython']
        shell = get_ipython().__class__.__name__  
        if shell in ['ZMQInteractiveShell','TerminalInteractiveShell','Shell']:
            return True   # Jupyter notebook or qtconsole or colab 
        else:
            return True  # Other type (?)
    except (NameError,KeyError) as e:
        return False      # Probably standard Python interpreter

def stop(msg='Stop here!'):
    raise Exception(msg)

def test(loader,
    model,
    opt = None,
    # info = None,
    save_dir = Path(''),
    testset = None,
    is_training = False,
    is_savecsv = False,
    criterion = None,
    optimizer = None,
    ):

    

    if is_training:
        device = next(model.parameters()).device  # get model device
        
        cols = ('','','val_loss','val_acc')
        # cols = ('val_loss','val_acc','','')
    else:
        logger.info("Predicting test dataset...")
        device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')

    running_loss = 0.0
    running_corrects = 0
    val_acc,val_loss = 0,0
    pred_list = []
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader,
            # file=sys.stdout,
            leave=True,
            bar_format='{l_bar}{bar:3}{r_bar}{bar:-10b}',
            total=len(loader), mininterval=1,)

        for i,data in enumerate(pbar,0):
            images, labels,_ = data
            images, labels  = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)  # predicted is a batch tensor result
            pred_list += preds.tolist()
            
            if is_training:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0) # losst.item * batch_size
                running_corrects += torch.sum(preds == labels.data)
                cols_str = ('%10s' * len(cols)) % (cols)
                pbar.set_description(cols_str, refresh=False)
       
        
        if is_training:
            val_loss = running_loss / len(loader.dataset)
            val_acc = running_corrects.double() / len(loader.dataset)
            s = ('%30.4g' + '%10.4g' * 1) % (val_loss,val_acc)
            # s = ('%10.4g' + '%10.4g' * 1) % (epoch_loss,epoch_acc)
            logger.info(s)
 

  
    # savecsv
    if is_savecsv:
        fn_list = loader.dataset.get_filenames()
        df = pd.DataFrame(columns=['filename','cid'])
        df['filename'] = fn_list
        df['cid'] = pred_list
        unified_fp = str(save_dir/'predictions.csv')
        df.to_csv(unified_fp, encoding='utf-8', index=False)
        logger.info('Done! Check csv: '+ unified_fp )

        # for _emoji only
        df = pd.read_csv('../02read/sample_submit.csv')
        csv_fp = str(save_dir/'emoji_submit.csv')
        map_fn2cid = dict(zip(fn_list, pred_list))
        # print(map_fn2cid)
        logger.info('Updaing _emoji df: '+ csv_fp )
        for i in tqdm(df.index):
            # print(i)
            fn = df.iloc[i]['name']
            cls_id =map_fn2cid[fn]
            df.at[i, 'label'] = loader.dataset.classes[cls_id]
        df.to_csv(csv_fp, encoding='utf-8', index=False)
        logger.info('done! check: '+ csv_fp )

    return pred_list,val_acc,val_loss


# rm
# test(testloader,model,testset=raw_test,is_savecsv=1,opt=opt)

# test(testloader,model,testset=raw_test,is_savecsv=1)

def infer(loader,model,classes,batch_index = 0, num = 4 ):
  # test images in loader
  dataiter = iter(loader)
  images, labels,_ = next(itertools.islice(dataiter,batch_index,None))
  images, labels  = images.to(device), labels.to(device)
  images = images[:num]

  # print images
  imshow(torchvision.utils.make_grid(images))

  outputs = model(images)
  _, predicted = torch.max(outputs, 1)
 
  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(len(images))))
  print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(images))))

  match_str =' '.join([('v' if (labels[j] == predicted[j] ) else 'x') 
              for j in range(len(images))])
  print('Result: ',match_str)

def show(loader,classes = None ,num=4, mean =(0.5,0.5,0.5),std=(0.5,0.5,0.5)):
    # preview train 
    # get some random training images
    classes = loader.dataset.classes if not classes else classes
    dataiter = iter(loader)
    images, labels, filenames = dataiter.next()
    images = images[:num]
    # show images
    imshow(torchvision.utils.make_grid(images),mean,std)
    print(' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
    


def imshow(img,mean =(0.5,0.5,0.5),std=(0.5,0.5,0.5)):
    # show img
    img = inverse_normalize(img,mean,std)     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def inverse_normalize(tensor, mean =(0.5,0.5,0.5),std=(0.5,0.5,0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

# %% dataset   
class Emoji2(VisionDataset):
    pkl_fp = '../03save.pkl'
    classes = ('angry', 'disgusted', 'fearful',
            'happy', 'neutral', 'sad', 'surprised')
    cls_names = classes


    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        super(Emoji2, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.train = train  # training set or test set

        self.data: Any = []
        self.targets = []

        # now load the pkl
        pkl_fp =  self.pkl_fp
        pkl_data = ax.load_obj(pkl_fp,silent=1)
        loaded_data = pkl_data['train_data'] if train else pkl_data['test_data']

        img_list, fn_list, label_id_list = [], [], []
        if self.train:
            for img_np, fn, labe_id in loaded_data:
                img_list += [img_np]
                fn_list += [fn]
                label_id_list += [labe_id]

        else:
            for img_np, fn in loaded_data:
                img_list += [img_np]
                fn_list += [fn]
                label_id_list += [0]

        img_np_list = np.asarray(img_list)  # convert to np
        img_np_list2 = np.repeat(
            img_np_list[:, :, :, np.newaxis], 3, axis=3)  # expand 1 channel to 3 channel 

        # print( img_np_list.shape)
        # print( 'img_np_list2 shape',img_np_list2.shape)

        
        # slice = 100
        # self.data = img_np_list2[:slice]
        # self.filenames = fn_list[:slice]
        # self.targets = label_id_list[:slice]

        self.data = img_np_list2
        self.filenames = fn_list
        self.targets = label_id_list


    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

    def get_filenames(self) -> list:
        return self.filenames

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target, fn = self.data[index], self.targets[index], self.filenames[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, fn


 

 

 
# infer(validloader,model,classes,3)




# %% set opt
# if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--args', nargs='?', const=True, default=False,help='load args from file ')
parser.add_argument('--weights', type=str, default='', help='initial weights path, override model')
parser.add_argument('--repro', action='store_true', help='only save final checkpoint')
# parser.add_argument('--alt_paras', action='store_true', help='change optimizer parameters')
parser.add_argument('--nopre', action='store_true', help='only save final checkpoint')
parser.add_argument('--freeze', action='store_true', help='only save final checkpoint')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--proxy', nargs='?', const=True, default=False, help='proxy')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--model', default='model_basic', help='set model when from scratch')
parser.add_argument('--kfold', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--skfold', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--seed', type=int, default='0', help='set seed for repro')
parser.add_argument('--split', type=float,   default='0.8', help='set seed for repro')
parser.add_argument('--data', type=str, default='Rawset', help='set seed for repro')

if isinteractive(): # not reliable, temp debug only  
    logger.info('[+]notebook \nNotebook mode ')
    opt = parser.parse_args(args=[]) 
else:
    logger.info('[+]bash \nBash mode')
    opt = parser.parse_args()
    


### opt explicit

# opt.model = 'res152_pre'
# opt.model = 'vgg11'
# opt.model = 'vgg19_bn' basic googlenet efficientnet-b0   res18
# opt.alt_paras = True  # only for freeze layers

opt.model = 'basic'
opt.epochs = 10
opt.batch = 32

opt.dataset = 'Emoji'   # Emoji Covid
opt.skfold = '1/5' 
opt.repro = True  
opt.nowandb = True
opt.project = '28emoji_lo'
opt.workers = 8
opt.seed = 0 
# opt.split = 0.8 # no-fold
# opt.freeze = True # need opt.model 
# opt.proxy = True
# opt.weights = '28emoji/exp21/weights/best.pt'
# opt.resume = '28emoji/exp38/weights/last.pt'
# opt.args = 'args.yaml'
# opt.notest = True
# opt.nosave = True
# opt.notest = True



# opt args 
if opt.args:
    with open(opt.args) as f:
        args = argparse.Namespace(**yaml.safe_load(f))  # replace
        logger.info('\n[+]args \nOverding args:')
        for k,v in opt.__dict__.items(): # ensure no new args
            if hasattr(args, k):
                v_overide =  getattr(args, k)  
                setattr(opt, k, v_overide)  # override
                logger.info('{}: {} > {}'.format(k,v,v_overide ))




# transform
train_mean, train_std = [0.5077, 0.5077, 0.5077], [0.2186, 0.2186, 0.2186]
test_mean, test_std = [0.5060, 0.5060, 0.5060], [0.2191, 0.2191, 0.2191]
transform_train = transforms.Compose(
    [transforms.ToTensor(), 
        transforms.Normalize(train_mean,train_std)])
transform_test = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)])


# opt extend 
opt.save_dir = str(increment_path(Path(opt.project) / opt.name))
# proxy
if opt.proxy:
    if opt.proxy == True:
        proxy_url = "http://127.0.0.1:1080" 
    else:
        proxy_url = opt.proxy
    print("\n[+]proxy \nProxy has been set to "+ proxy_url)
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url


# Reproducibility,  NOT work in notebook!
seed = random.randint(0,9999)
if opt.repro:
    logger.info('\n[+]repro')
    seed = opt.seed
    os.environ['ICH_REPRO'] = '1'
    os.environ['ICH_SEED'] = str(seed)
    import models  # must after os.emviron is defined
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    if isinteractive(): 
        logger.info(' U R in interative, torch.use_deterministic_algorithms set False !')
        torch.use_deterministic_algorithms(False)
    logger.info('Enabled, seed: {}, ICH_REPRO: {}'.format(os.environ['ICH_SEED'],os.environ['ICH_REPRO']))
else: 
    os.environ['ICH_REPRO'] = '0'
    os.environ['ICH_SEED'] = str(seed)
    import models # must after os.emviron is defined
    torch.use_deterministic_algorithms(False) # for notenook!
    g = torch.Generator()
    g.manual_seed(seed)
 

# resume
if opt.resume:
    ckpt = opt.resume if isinstance(opt.resume, str) else None  # ckpt is str(temp.pt)
    assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
    with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.weights = ckpt
        print('Resuming training from %s' % ckpt)
        temp_dict = torch.load(opt.weights)
        resume_wandb_id =  temp_dict['wandb_id']
        del temp_dict
    if not resume_wandb_id:
        raise Exception("No need to resume.")


# load opt, after resume 
weights = opt.weights
split_dot = opt.split   
workers = opt.workers
batch_size = opt.batch
epochs = opt.epochs
save_dir = Path(opt.save_dir)
wdir = save_dir / 'weights'
# Directories
wdir.mkdir(parents=True, exist_ok=True)  # make dir
last = wdir / 'last.pt'
best = wdir / 'best.pt'
results_file = save_dir / 'results.txt'
logger.addHandler(logging.FileHandler(save_dir / 'logger.txt')) 
logger.info('\n[+]opt\n' + json.dumps(opt.__dict__, sort_keys=True) )
# Save run settings
# with open(save_dir / 'hyp.yaml', 'w') as f:
#     yaml.safe_dump(hyp, f, sort_keys=False)
with open(save_dir / 'opt.yaml', 'w') as f:
    yaml.safe_dump(vars(opt), f, sort_keys=False)


# gpu  
msg = "\n[+]device \nICH 🚀 v0.1 Using device: {}".format(device) 
#Additional Info when using cuda
if device.type == 'cuda':
    msg = msg + " ["
    msg = msg + torch.cuda.get_device_name(0)
    msg +=  ']\nGPU mem_allocated: {}GB, cached: {}GB'.format(
        round(torch.cuda.memory_allocated(0)/1024**3,1),
        round(torch.cuda.memory_reserved(0)/1024**3,1),
    ) 
logger.info(msg)

# clean tqdm for notebook
try:
    tqdm._instances.clear()
except Exception:
    pass




# dataset
logger.info('\n[+]dataset')
# %% kfold 
import datasets

rawset =  getattr(datasets, opt.dataset)
raw_train = rawset(root='./input', train=True,
                    transform=transform_train)
raw_test = rawset(root='./input', train=False,
                    transform=transform_test)

# from datasets import Emoji
# raw_train = Emoji(root='./data', train=True,
#                     transform=transform_train)
# raw_test = Emoji(root='./data', train=False,
#                     transform=transform_test)


classes = raw_train.classes
nc = len(classes)
if opt.kfold or opt.skfold:  # fold, will ignore split_dot
    fold_str = opt.kfold if opt.kfold else opt.skfold
    nf, cf = int(fold_str.split('/')[1]),int(fold_str.split('/')[0]) # num-fold,current-fold
    if opt.kfold:
        logger.info('KFold: '+ fold_str)
        fold_obj = KFold(n_splits=nf, random_state=seed, shuffle=True)
    elif opt.skfold:
        logger.info('StratifiedKFold: ' + fold_str)
        fold_obj = StratifiedKFold(n_splits=nf, random_state=seed, shuffle=True)

    for i_fold, ids in enumerate(fold_obj.split(raw_train,raw_train.targets)):
        if (i_fold + 1) == cf:
            trainset = torch.utils.data.Subset(raw_train,ids[0])
            validset = torch.utils.data.Subset(raw_train,ids[1])
            break
else: # nofold, use split_dot
    num_train = int(len(raw_train) * split_dot)
    trainset, validset = random_split(raw_train,
                                    [num_train, len(raw_train) - num_train],                                    
                                    generator=g)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers,
                                            worker_init_fn=seed_worker,
                                            generator=g,)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                        shuffle=False, num_workers=workers,
                                        worker_init_fn=seed_worker,
                                        generator=g)
testloader = torch.utils.data.DataLoader(raw_test, batch_size=batch_size,
                                        shuffle=False, num_workers=workers,
                                        worker_init_fn=seed_worker,
                                        generator=g)
logger.info("Dataset loaded: " + opt.dataset)


# model
logger.info('\n[+]model')
pretrained = weights.endswith('.pt')
if pretrained:
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    model_name = ckpt['model_name']
    model = models.get(model_name,nc,opt.freeze,opt.nopre)
    model.load_state_dict(state_dict, strict=False)  # load
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
else:
    model_name = opt.model
    model = models.get(model_name,nc,opt.freeze,opt.nopre)
model.to(device)
logger.info("Loaded modle: "+ model_name)

# hyp 
criterion = nn.CrossEntropyLoss()
params_to_update = model.parameters()
need_learn = ''
if opt.freeze:
    logger.info("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            logger.info("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            need_learn += name + ', '
    # logger.info("Params to learn: "+ need_learn) # too many info
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
start_epoch, best_acc = 0, 0.0
if pretrained:
    # Optimizer
    if ckpt['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_acc = ckpt['best_acc']
    # Results
    if ckpt.get('training_results') is not None:
        results_file.write_text(ckpt['training_results'])  # write results.txt
    # Epochs
    start_epoch = ckpt['epoch'] + 1
    if opt.resume:
        assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
    if epochs < start_epoch: # will never run cause strip_opti set to -1
        logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                    (weights, ckpt['epoch'], epochs))
        epochs += ckpt['epoch']  # finetune additional epochs
    del ckpt, state_dict
scheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=2)
scheduler.last_epoch = start_epoch - 1  # do not move



# visual init
logger.info('\n[+]visual')
# Tensorboard
prefix = colorstr('tensorboard: ')
logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
writer = SummaryWriter(opt.save_dir)  # Tensorboard
# wandb
# if 'wandb' in vars() or 'wandb' in globals():  wandb.finish() # for notebook only
if not opt.nowandb:
    logger.info("Check is wandb installed ...")
    name_inc = os.path.normpath(opt.save_dir).split(os.path.sep)[-1]
    wandb = util.wandb
    if wandb:
        logger.info("Wandb login ...")
        is_wandb_login = wandb.login()
        # init wandb               
        wandb.init(
            # Set entity to specify your username or team name
            # ex: entity="carey",
            # Set the project where this run will be logged
            project=Path(opt.project).stem, 
            # Track hyperparameters and run metadata
            config=dict(opt.__dict__),
            id = torch.load(weights).get('wandb_id') if opt.resume else wandb.util.generate_id(),
            name=name_inc,
            resume="allow",
            tags = ["explorer"],
        )
    else:
        logger.info('Wandb not installed, manual install: pip install wandb')
else:
    wandb = None
# visual images in 1 batch 
images, labels, _ = next(iter(trainloader))
images, labels = images.to(device), labels.to(device)
images = inverse_normalize(images,train_mean,train_std)
grid = torchvision.utils.make_grid(images)
writer.add_image('train_images', grid, 0)
# visual model 
try:
    writer.add_graph(model, images)
except Exception as e:
    logger.info(e)
# visual cls distri 
plot_cls_bar(raw_train.targets, save_dir, raw_train)
# writer.add_image('cls_distri_img', cls_distri_img, 0)
# writer.add_histogram('raw_train_classes', torch.tensor(raw_train.targets), 1)
# writer.add_histogram('raw_test_classes', torch.tensor(raw_test.targets), 1)

# visual info 
logger.info("\n[+]info")
dataset_msg = 'No msg'
if opt.kfold:
    dataset_msg = "Kfold: " + opt.kfold
elif opt.skfold:
    dataset_msg = "StratifiedKFold: " + opt.skfold
else:
    dataset_msg = "Split_dot: " + str(split_dot)
summary_str = "- dataset: {}".format(dataset_msg)
summary_str += "  raw_train:{},raw_test:{}, trainset:{}, validset:{}, nc:{}, batch:{}".format(
     len(raw_train),len(raw_test),len(trainset),len(validset),len(classes),batch_size
)
# logger.info(model)
logger.info(summary_str)
writer.add_text('summary', summary_str, 0)
writer.add_text('opt', str(opt.__dict__), 0)

# visual model in bash 
# input_shape = next(iter(trainloader))[0][0].shape
# print("Input shape:",input_shape)
# summary(model, input_shape)

# Start Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
logger.info('\n[+]train')
logger.info('Logging results to ' + str(save_dir))
logger.info('Starting training for {} epochs...'.format(epochs))
best_model_wts = copy.deepcopy(model.state_dict())
since = time.time()
for epoch in range(start_epoch,epochs):
    logging.info("")
    final_epoch = epoch + 1 == epochs

    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    cols = ('Epoch','gpu_mem','tra_loss','train_acc')
    logger.info(('%10s' * len(cols)) % (cols))

    pbar = tqdm(trainloader,
        # file=sys.stdout,
        leave=True,
        bar_format='{l_bar}{bar:3}{r_bar}{bar:-10b}',
        total=len(trainloader), mininterval=1,
    )
    for i,(inputs, labels, _) in  enumerate(pbar,0):
        inputs,labels  = inputs.to(device),labels.to(device)

        # core
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward + backward + optimize
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
        mloss = running_loss / (i * batch_size + inputs.size(0))
        macc = running_corrects / (i * batch_size + inputs.size(0))
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' * 2 + '%10.4g' * 2) % (
            '%g/%g' % (epoch, epochs - 1), mem ,mloss,macc    )
        pbar.set_description(s, refresh=False)
        imgs = inputs
        # writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
        # end batch  -----------------
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = running_corrects.double() / len(trainloader.dataset)
    scheduler.step()
    
    # Validate
    this_best = False
    val_acc = 0 
    val_loss = 0
    if (opt.split != 1) and (not opt.notest or final_epoch):
        _pred_list,val_acc,val_loss = test(validloader,
            model,
            is_training = True,
            criterion = criterion,
            optimizer = optimizer,
            )

    if val_acc > best_acc:
        best_acc = val_acc
        best_model = copy.deepcopy(model)
        best_model_wts = copy.deepcopy(model.state_dict())
        this_best = True
    
    # Log
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Acc/train', epoch_acc, epoch)
    writer.add_scalar('Acc/val', val_acc, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    if wandb:
        wandb.log({
            "train_loss":epoch_loss,
            "val_loss":val_loss,
            "train_acc":epoch_acc,
            "val_acc":val_acc,
            "lr":optimizer.param_groups[0]['lr'],
        })
        
    # Save
    if (not opt.nosave) or (final_epoch and not opt.evolve):
        ckpt = {
            'model_name' : opt.model,
            'epoch': epoch,
            'model': deepcopy(model).half(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'best_acc': best_acc,
            'wandb_id': wandb.run.id if wandb else None,
        }
        torch.save(ckpt, last)
        if this_best:
            torch.save(ckpt, best)  
        del ckpt
    # end epoch -----------------------------

# summary  
time_elapsed = time.time() - since
msg_sum = '{} epochs completed in {:.0f}m {:.0f}s \n\n'.format(
    epochs - start_epoch , time_elapsed // 60, time_elapsed % 60) # epoch should + 1?
msg_sum += 'Best val Acc: {:4f}'.format(best_acc)
logger.info(msg_sum) 
writer.add_text('msg_sum', msg_sum, 0)



# Strip optimizers
final = best if best.exists() else last  # final model
for f in last,best:
    if f.exists():
        f_striped = f.with_suffix('.strip.pt')
        copyfile(f, f_striped)
        strip_optimizer(f_striped)  # strip optimizers


# release resource
writer.close()
if wandb:
    wandb.summary['best_val_acc'] = best_acc
    wandb.finish()
torch.cuda.empty_cache()

logger.info('End Train!')


# %% test
logger.info('\n[+]test')
logger.info('loading best model ')
model.load_state_dict(best_model_wts)
# will error as sliced
test(testloader,model,testset=raw_test,is_savecsv=1,opt=opt,save_dir = save_dir) 

logger.info('End Test!')
