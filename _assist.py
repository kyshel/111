# snippets that assist devleop
# type: ignore
# flake8: noqa

# %%
# %%
# %%
# %% 




a = [1,2]
b,c = a
print(b,c)




exit()

import ax
from pathlib import Path

a = 123
fp  = '_rm6666/_rm5555555/_rm2222.txt'


ax.mkdir(Path(fp).parent)

exit()



# %% calc dataset mean&std
from tqdm import tqdm 
import torch

def get_mean_and_std2(dataset,batch=1):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch )
 
    mean = 0.0
    for images, _ in tqdm(loader):
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*224*224))


  
    print(mean,std)



def get_mean_and_std(dataset,batch=1):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch,)
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(tqdm(loader),0):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)
    print(mean,std)

 
transform = transforms.Compose([transforms.Resize([2000,2000]),
    transforms.ToTensor(),])

from datasets import Covid
d1 = Covid(
        root='../03png/train', 
        obj = '../11data/train_imginfo.obj',
        train=True,
        transform=transform,
        workers = opt.workers,
        )

d2 = Covid(
        root='../03png/test', 
        obj = '../11data/test_imginfo.obj',
        train=False,                    
        transform=transform ,
        workers = opt.workers,
        
        )


get_mean_and_std(d1,batch = opt.batch)
get_mean_and_std(d2,batch = opt.batch)













# %%



import numpy as np


np.random.RandomState(0)







