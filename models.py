
import torch.nn as nn
import torch
from torch.nn import parameter
import torch.nn.functional as F
import torchvision

import os 
import numpy as np
 
# repro 
repro_flag = "ICH_REPRO"

if repro_flag not in os.environ:
    raise Exception('Please set repro flag: '+repro_flag)

if os.environ[repro_flag] == '1':
    seed = 0
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)

 
opti_paras = []

def get(model_name ='basic'):
    if model_name not in globals():
        raise Exception( model_name + " model not found, please check valid models in  models.py")
    return globals()[model_name]()

 
def basic():
    return  nn.Sequential( # > 3 48 48
                nn.Conv2d(3,6,5), # > 6 44 44
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 6 22 22

                nn.Conv2d(6,16,5), # 16 18 18
                nn.ReLU(), 
                nn.MaxPool2d(2, 2), # 16 9 9

                nn.Flatten(), # 16*9*9 

                nn.Linear(16*9*9, 256), 
                nn.ReLU(),

                nn.Linear(256, 7),
    )



 

def basic_overfit():
    return nn.Sequential( # > 3 48 48
            nn.Conv2d(3,6,5), # > 6 44 44
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 6 22 22

            nn.Conv2d(6,16,5), # 16 18 18
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), # 16 9 9


            nn.Flatten(), # 16*9*9 

            nn.Linear(16*9*9, 2048), 
            nn.ReLU(),


            nn.Linear(2048, 2048),
            nn.ReLU(),


            nn.Linear(2048, 2048),
            nn.ReLU(),

            nn.Linear(2048, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),


            nn.Linear(128, 7),
)




def res18():
    m = torchvision.models.resnet18()
    m.fc = nn.Linear(res18.fc.in_features, 7)
    return m

def res18_pre():
    m  = torchvision.models.resnet18(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, 7)
    return m

def res18_pre_fre():
    global opti_paras
    m  = torchvision.models.resnet18(pretrained=True)
    for param in m.parameters():
        param.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 7)
    opti_paras['res18_pre_fre'] = m.fc.parameters()
    return m


def res34_pre():
    m  = torchvision.models.resnet34(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, 7)
    return m

def res152_pre():
    m  = torchvision.models.resnet152(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, 7)
    return m












 


        