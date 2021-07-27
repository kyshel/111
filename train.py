
# %% introduction
# ich means Image Classification Hammer, used for imgae classification task.
# Developed in vscode with interactive features

'''  
# p1
- add checkpoint 


# p9
- unify data


# logs
- v6 save last.pt and best.pt
'''
# %% preset



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
reload(util)
reload(ax)
import  itertools 
import sys
import logging

logging.basicConfig(
        format="%(message)s",
        level=logging.INFO)
logger = logging.getLogger(__name__)


# %% functions



def stop(msg='Stop here!'):
    raise Exception(msg)

def test(loader,
    model,
    testset = None,
    is_training = 0,
    is_savecsv = 0):
    if is_training:
        pass
    else:
        logger.info("Predicting test dataset...")

    pred_list = []
 

    model.eval()
    with torch.no_grad():
        for data in tqdm(loader):
            images, labels,_ = data
            images, labels  = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # predicted is a batch tensor result
            pred_list += predicted.tolist()


    if is_savecsv:
        fn_list = testset.get_filenames()
        df = pd.DataFrame(columns=['filename','cid'])
        df['filename'] = fn_list
        df['cid'] = pred_list
        unified_fp = './runs_t/prediction_{}.csv'.format(ax.nowtime())
        df.to_csv(unified_fp, encoding='utf-8', index=False)
        logger.info('Done! Check csv: '+ unified_fp )

        # for _emoji only
        csv_fp = "./runs_t/s_{}.csv".format(ax.nowtime())
        map_fn2cid = dict(zip(fn_list, pred_list))
        df = pd.read_csv('/content/02read/sample_submit.csv')
        logger.info('Updaing _emoji df: '+ csv_fp )
        for i in tqdm(df.index):
            # print(i)
            fn = df.iloc[i]['name']
            cls_id =map_fn2cid[fn]
            df.at[i, 'label'] = classes[cls_id]


        df.to_csv(csv_fp, encoding='utf-8', index=False)
        logger.info('done! check: '+ csv_fp )



    return pred_list

# rm
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

def show(loader,classes,num=4):
    # preview train 
    # get some random training images
    dataiter = iter(loader)
    images, labels, filenames = dataiter.next()
    images = images[:num]
    # show images
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
    

def imshow(img):
    # show img
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %% dataset   
class Emoji(VisionDataset):
    pkl_fp = '/content/_emoji/03save.pkl'
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

        super(Emoji, self).__init__(root, transform=transform,
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
            img_np_list[:, :, :, np.newaxis], 3, axis=3)  # expand 1 axis

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



# %% model  
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(1296, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        show_list = []
 
        show_list += [x.shape]
        # print(show_list)
        x = self.pool(F.relu(self.conv1(x)))
        
        show_list += [x.shape]
        x = self.pool(F.relu(self.conv2(x)))
        show_list += [x.shape]
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        show_list += [x.shape]
        x = F.relu(self.fc1(x))
        show_list += [x.shape]
        x = F.relu(self.fc2(x))
        show_list += [x.shape]
        x = self.fc3(x)
        show_list += [x.shape]
        for i,v in enumerate(show_list):
          # print(i,v)
          pass
         
        return x


# %% main

# mv to top at last 
reload(util)
from util import increment_path
from pathlib import Path

# begin
parser = argparse.ArgumentParser()
parser.add_argument('--placeholder', type=str,
                    default='blank', help='initial weights path')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
opt = parser.parse_args(args=[])

# default opt
opt.nosave = False
opt.notest = False
opt.evolve = False
opt.project = 'runs_train'
opt.name = 'exp'

# explicit opt
opt.batch = 512
opt.split = 0.8
opt.workers = 2
opt.epochs = 3
opt.save_dir = str(increment_path(Path(opt.project) / opt.name))



logger.info('[+]opt\n' + json.dumps(opt.__dict__, sort_keys=True) )
split_dot = opt.split  
workers = opt.workers
batch_size = opt.batch
epochs = opt.epochs
save_dir = opt.save_dir
print('save_dir',save_dir)
wdir = save_dir / 'weights'

wdir.mkdir(parents=True, exist_ok=True)  # make dir
last = wdir / 'last.pt'
best = wdir / 'best.pt'
results_file = save_dir / 'results.txt'

# clean tqdm 
try:
    tqdm._instances.clear()
except Exception:
    pass



# GPU info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
msg = "\n[+]device \nICH 🚀 v0.1 Using device: {}".format(device) 
#Additional Info when using cuda
if device.type == 'cuda':
    msg = msg + " + "
    msg = msg + torch.cuda.get_device_name(0)
    msg +=  '\nGPU mem_allocated: {}GB, cached: {}GB'.format(
        round(torch.cuda.memory_allocated(0)/1024**3,1),
        round(torch.cuda.memory_reserved(0)/1024**3,1),
    ) 

logger.info(msg)



# Prepare datasets
logger.info('\n[+]load')
transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
raw_train = Emoji(root='./data', train=True,
                    transform=transform)
raw_test = Emoji(root='./data', train=False,
                    transform=transform)
classes = raw_train.classes
num_train = int(len(raw_train) * split_dot)
trainset, validset = \
    random_split(raw_train, [num_train, len(raw_train) - num_train],
                    generator=torch.Generator().manual_seed(42))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                        shuffle=False, num_workers=workers)
testloader = torch.utils.data.DataLoader(raw_test, batch_size=batch_size,
                                        shuffle=False, num_workers=workers)

dataset_sizes ={'train':len(trainset),"val":len(validset)}
logger.info("Dataset info > split_dot:{}, train/test={}/{}, classes_count: {}, batch_size:{}".format(
    split_dot,len(raw_train),len(raw_test),len(classes),batch_size
))
logger.info("Dataset loaded.")

# Prepare Model
model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
logger.info('Model loaded.')




# train
logger.info('\n[+]train')
logger.info('Starting training for {} epochs...'.format(epochs))
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
for epoch in range(epochs):
    logging.info("")
    final_epoch = epoch + 1 == epochs
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        loader = trainloader if phase is 'train' else validloader
        
        cols = ('Epoch','gpu_mem','tra_loss','train_acc')
        if phase == 'train':
            logger.info(('%10s' * len(cols)) % (cols))
        else:
            cols = ('','','val_loss','val_acc')
        

        pbar = tqdm(loader,
            # file=sys.stdout,
            leave=True,
            bar_format='{l_bar}{bar:3}{r_bar}{bar:-10b}',
            total=len(loader), mininterval=1,)
        for i,(inputs, labels, _) in  enumerate(pbar,0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
            mloss = running_loss / (i * batch_size + inputs.size(0))
            macc = running_corrects / (i * batch_size + inputs.size(0))
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            # s = str((epoch, epochs - 1, mem,  labels.shape[0], inputs.shape[-1]))
             
            
            if phase == 'train':
                s = ('%10s' * 2 + '%10.4g' * 2) % (
                    '%g/%g' % (epoch, epochs - 1), mem ,mloss,macc    )
                pbar.set_description(s, refresh=False)
                
            else:
                cols_str = ('%10s' * len(cols)) % (cols)
                pbar.set_description(cols_str, refresh=False)
                s = ('%30.4g' + '%10.4g' * 1) % (mloss,macc)
        
        

            

            # end batch  -----------------

        if phase == 'train':
            scheduler.step()
        else:
            logging.info(s)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        # logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #         phase, epoch_loss, epoch_acc))
 
        # update best val_acc
        this_best = False
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            this_best = True

        if (not opt.nosave) or (final_epoch and not opt.evolve):
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(ckpt, last)
            if this_best:
                torch.save(ckpt, best)  
            del ckpt

        # end phase ------------------------

    
 
    # end epoch -----------------------------
time_elapsed = time.time() - since
print('{} epochs complete in {:.0f}m {:.0f}s \n'.format(
    epochs, time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))



# Save model
best_fp = './runs_t/v6_best_{}.pth'.format(ax.nowtime())
last_fp = './runs_t/v6_last_{}.pth'.format(ax.nowtime())
model.to('cpu')
torch.save(best_model_wts, best_fp)
torch.save(model.state_dict(), last_fp)
model.to(device)
print('Model saved to ', best_fp)
print('Model saved to ', last_fp)



# %% test
logger.info('\n[+]test')
model.load_state_dict(best_model_wts)
test(testloader,model,testset=raw_test,is_savecsv=1)

logger.info('end')

 

 
#%% exp

util.rm1()

# infer(validloader,model,classes,3)









 

# %% set opt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--placeholder', type=str,
                        default='blank', help='initial weights path')
    opt = parser.parse_args(args=[])

    print(json.dumps(opt.__dict__, sort_keys=True))

