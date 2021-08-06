# datasets

# preset
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


 

class Rawset(VisionDataset):
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
                 root =  '03png/train' ,
                 csv = 'train_study_level.csv',
                 obj = '11data/train_imginfo.obj',
                 train=True, transform=None, ):
        """
        Args:
            root
        """
        self.csv = csv
        self.obj = obj
        self.root = root     
        self.transform = transform
        self.train = train   # trainset or testset


        self.img2cat_list = self.get_img2cat_list()   

    def __len__(self):
        return len(self.img2cat_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        catid = self.img2cat_list[idx][1] 
        imgid = self.img2cat_list[idx][0]

        fn = imgid + '.png'
        fp = os.path.join(self.root,  fn)
        im = Image.open(fp).convert('RGB')
        # pix = np.array(im)
        pix = im

        if self.transform:
            pix = self.transform(pix)
 
        return pix,catid,imgid

    def get_img2cat_list(self):
        # snip, load img2cat_list 
        csv = self.csv
        obj = self.obj

        df = pd.read_csv(csv)

        study2cat_map={}
        for i, row in df.iterrows():
            for i_col in range(1,5):
                if row[i_col] == 1:
                    study2cat_map[ row[0].split('_')[0]] = i_col - 1

        imgdict = ax.load_obj(obj,silent = True)
        img2cat_list = []
        for k,v in imgdict.items():
            study = v['study_id']
            if self.train:
                img2cat_list += [[k,study2cat_map[study] ]]
            else:
                img2cat_list += [[k,-1 ]]

    
        return img2cat_list

 
