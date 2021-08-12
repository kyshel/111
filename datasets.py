# datasets

# %% datasets
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
from PIL import Image, ExifTags
from torch.utils.data import Dataset
import pandas as pd
from multiprocessing.pool import ThreadPool
from itertools import repeat
from tqdm import tqdm 
import sys
from pathlib import Path
import glob
import logging
import argparse
import yaml 
from pprint import pprint 
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


# logger 
for handler in logging.root.handlers[:]:  
    # https://stackoverflow.com/questions/12158048/changing-loggings-basicconfig-which-is-already-set
    logging.root.removeHandler(handler)
logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            # logging.FileHandler(save_dir / 'logger.txt'), # should get dir after 
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)
# logger.addHandler(logging.FileHandler(f'_logger_{os.path.basename(__file__)}.txt')) 


# repro >>> start
repro_flag = "ICH_REPRO"
repro_seed = "ICH_SEED"
# if repro_flag not in os.environ:
#     raise Exception('Please set repro flag: '+repro_flag)
if (repro_flag in os.environ) and (repro_seed in os.environ):
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
else:
    logger.info("Not repro!")
# repro <<< end 

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break 


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s



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


 


class LoadImageAndLabels(VisionDataset):  # delete 
    _repr_indent = 4

    def __init__(
            self,
            opt,
            train = True,
            logger = logger,
            # root: str,
            # transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            prefix ='',
            workers = 8,
            cache_images = None, 

        ) -> None:

        split = 'train' if train else 'test'
        self.split = split
        self.cache_images = cache_images
        self.transform = transform

        # get data.yaml info 
        with open(opt.data) as f:
            opt_data = argparse.Namespace(**yaml.safe_load(f))  # replace
        
        src = Path(opt_data.src)
        nc = len(opt_data.names)
        csv_fp = Path(opt_data.src) / 'labels.csv'
        self.names = opt_data.names

        # download if not exist
        if  not os.path.isdir(opt_data.src) and hasattr(opt_data, 'download')   :
            download_and_extract_archive(
                url = opt_data.download,
                download_root = src.parent ,
                )
        

        # make pid2cid map from labels.csv
        df = pd.read_csv(csv_fp,converters={0:str,1:int})
        images = df[df.columns[0]].astype(str).tolist()
        labels = df[df.columns[1]].astype(int).tolist()
        image2label = dict(zip(images, labels))
 
        # make files list from images
        # f = []
        files = {}
        files = sorted(glob.glob( str(src / split / '*')  ))
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
   
        # make info 
        info = []  # final valid contariner
        nf,nm,nc = 0,0,0
        pbar = tqdm(images)
        for fp in pbar:
            fo = Path(fp)
            try:
                # check imgs
                im = Image.open(fp)
                im.verify()
                shape = exif_size(im)  # image size 

                image_f = fo.stem # {image_f}.png
                if image_f in image2label:          
                    info += [[image_f, image2label[image_f], fp] ]  # image, label, path
                    nf += 1  # image found label
                else:                               
                    info += [[image_f, 0, fp]]
                    nm += 1  # image missing label
                    
            except Exception as e:
                nc += 1 # image corrupted 
                logger.info(f'{prefix}WARNING: Ignoring corrupted image and/or label {f}: {e}')

            # ns = len(pbar) - nf - nm - nc  # image shortaged
            pbar.desc = f"{prefix}Scanning '{src}' images and labels... " \
                        f"{nf} found, {nm} miss, {nc} corrupted "
        pbar.close()
        
        self.info = info
        self.images = [i[0] for i in info]
        self.labels = [i[1] for i in info]
        self.paths = [i[2] for i in info]

        n = len(self.info)
        self.pixs = [None] * n

        # make cache 
        if self.cache_images:
            gb=0
            ids = range(n)
            results = ThreadPool(workers).imap(self.load_pix,  ids  )  
            pbar = tqdm(enumerate(results), total=n, mininterval=1,)
            for i, x in pbar:
                self.pixs[i] = x  
                # gb += self.pixs[i].nbytes # sys.getsizeof(b.storage())
                gb +=  sys.getsizeof(self.pixs[i].storage())
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()


        



    def __getitem__(self, index: int) -> Any:
        label = self.labels[index]
        path = self.paths[index]
        if self.cache_images:
            return self.pixs[index], label, path # pix, label, path
        else:
            return self.load_pix(index), label, path 
        


    def __len__(self) -> int:
        return len(self.info)

    def load_pix(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pix = Image.open(self.paths[idx]).convert('RGB')
        if self.transform: pix = self.transform(pix)
        return pix  




    

if __name__ == '__main__':
    logger.info("Running directly...")
 
    yaml_fp = '27bra_lo/exp5/opt.yaml'
    with open(yaml_fp) as f:
        opt = argparse.Namespace(**yaml.safe_load(f))  # replace


        # print(opt)
        aaa = LoadImageAndLabels(opt,train = True)
        bbb = LoadImageAndLabels(opt,train = False)




# %%
