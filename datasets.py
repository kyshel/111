# datasets

# preset
from packaging import version
from torchvision.datasets.vision import VisionDataset
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
from torchvision import models 
import os 
import numpy as np
from typing import Any, Callable, Optional, Tuple
import ax
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from multiprocessing.pool import ThreadPool
from itertools import repeat
from tqdm import tqdm 
import sys
from pathlib import Path


# repro >>> start
repro_flag = "ICH_REPRO"
repro_seed = "ICH_SEED"
if repro_flag not in os.environ:
    raise Exception('Please set repro flag: '+repro_flag)

seed = int(os.environ[repro_seed])
if os.environ[repro_flag] == '1':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    np.random.seed(seed)
    if version.parse(torch.__version__) >= version.parse("1.8.0"):
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
# repro <<< end 


# Emoji
class Emoji(VisionDataset):
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


 

class Rawset(VisionDataset):  # delete 
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
         pass

    def __getitem__(self, index: int) -> Any:
         pass

    def __len__(self) -> int:
         pass

 


class Covid(Dataset):
    """Face Landmarks dataset."""

    cats = ['0neg','1typical','2indeter','3atypical']
    classes = cats

    def __init__(self, 
                 root,  # '../03png/train'
                 obj,   #  '../11data/train_imginfo.obj'
                 csv = '../00raw/train_study_level.csv', 
                 train=True, transform=None,cache_images=None,workers=2,prefix='' ):
        """
        Args:
            root
        """

        self.csv = csv
        self.obj = obj
        self.root = root     
        self.transform = transform
        self.train = train   # trainset or testset
        self.cache_images = cache_images
        self.pid2cid = self.get_pid2cid()   

        
        targets = []
        filenames = []
        for pid,cid in self.pid2cid:
            targets+= [cid]
            filenames += [pid]
        self.targets = targets
        self.filenames = filenames

        n = len(self.pid2cid)
        self.pixs = [None] * n
        self.cids = [None] * n
        self.pids = [None] * n

        if self.cache_images:
            gb=0
            ids = range(n)
            results = ThreadPool(workers).imap(self.load_img,  ids  )  # 8 threads
            pbar = tqdm(enumerate(results), total=n, mininterval=1,)
            for i, x in pbar:
                self.pixs[i],self.cids[i],self.pids[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                # gb += self.pixs[i].nbytes # sys.getsizeof(b.storage())
                gb +=  sys.getsizeof(self.pixs[i].storage())
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()


        

    def __len__(self):
        return len(self.pid2cid)

    def __getitem__(self, idx):
        if self.cache_images:
            i = idx
            return self.pixs[i],self.cids[i],self.pids[i]
        else:
            return self.load_img(idx) # pix,cid,pid
             

    def load_img(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        cid = self.pid2cid[idx][1] 
        pid = self.pid2cid[idx][0]

        fn = pid + '.png'
        fp = os.path.join(self.root,  fn)
        im = Image.open(fp).convert('RGB')
        # pix = np.array(im)
        pix = im

        if self.transform:
            pix = self.transform(pix)

        return pix,cid,pid

    def get_pid2cid(self): # [['pic1',1],['pic2',2]]
        # snip, load pid2cid 
        csv = self.csv
        obj = self.obj

        df = pd.read_csv(csv)

        study2cat_map={}
        for i, row in df.iterrows():
            for i_col in range(1,5):
                if row[i_col] == 1:
                    study2cat_map[ row[0].split('_')[0]] = i_col - 1

        imgdict = ax.load_obj(obj,silent = True)
        pid2cid = []
        for k,v in imgdict.items():
            study = v['study_id']
            if self.train:
                pid2cid += [[k,study2cat_map[study] ]]
            else:
                pid2cid += [[k,-1 ]]

    
        return pid2cid

 
