
import torch.nn as nn
import torch
from torch.nn import parameter
import torch.nn.functional as F
from torchvision import models 

import os 
import numpy as np
 
env_inspect = "ECH_INSPECT"

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

 
# stale >>> 
opti_paras = []

def get2(model_name ='basic'):
    if model_name not in globals():
        raise Exception( model_name + " model not found, please check valid models in  models.py")
    return globals()[model_name]()

 
# def basic():
#     return  nn.Sequential( # > 3 48 48
#                 nn.Conv2d(3,6,5), # > 6 44 44
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 2), # 6 22 22

#                 nn.Conv2d(6,16,5), # 16 18 18
#                 nn.ReLU(), 
#                 nn.MaxPool2d(2, 2), # 16 9 9

#                 nn.Flatten(), # 16*9*9 
#                 # print(1111111111111),

#                 # nn.Linear(16*9*9, 256), 
#                 nn.Linear(394384, 256), 
#                 nn.ReLU(),

#                 nn.Linear(256, 7),
#     )



 

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
    m = models.resnet18()
    m.fc = nn.Linear(res18.fc.in_features, 7)
    return m

def res18_pre():
    m  = models.resnet18(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, 7)
    return m

def res18_pre_fre():
    global opti_paras
    m  = models.resnet18(pretrained=True)
    for param in m.parameters():
        param.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 7)
    opti_paras['res18_pre_fre'] = m.fc.parameters()
    return m


def res34_pre():
    m  = models.resnet34(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, 7)
    return m

def res152_pre():
    m  = models.resnet152(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, 7)
    return m

def vgg11():
    m  = models.vgg11(pretrained=True )
    m.classifier[6] = nn.Linear(4096,7)
    return m

def vgg19():
    m  = models.vgg19(pretrained=True )
    m.classifier[6] = nn.Linear(4096,7)
    return m


def vgg19_bn():
    m  = models.vgg19_bn(pretrained=True )
    m.classifier[6] = nn.Linear(4096,7)
    return m

# stale <<<<<<<

class Net(nn.Module):  
    #  not runnable, rm
    def __init__(self,nc):
        super().__init__()

        self.nc = nc
        self.linear_input_size = 16*9*9

        self.conv_net = nn.Sequential(
            nn.Conv2d(3,6,5), # > 6 44 44
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 6 22 22

            nn.Conv2d(6,16,5), # 16 18 18
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), # 16 9 9

            nn.Flatten(), # 16*9*9 
        )

        self.fc =  nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, self.nc),
        )
    def forward(self, x):
        x = self.conv_net(x)
        # print(x.size())
        x = self.fc(x)
        return x
 



class PrintLayer(nn.Module):   
    # https://discuss.pytorch.org/t/how-do-i-print-output-of-each-layer-in-sequential/5773/4
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        if os.environ[env_inspect] == '1':
            print(">>>Before linear:",x.size())
        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    if feature_extract:
        print("Freeze enbled!")
    
    model_name = 'efficientnet-b0' if model_name in ['efb0'] else model_name

    if model_name == "basic":
        before_liner = 2704
        model_ft = nn.Sequential( # > 3 48 48
                nn.Conv2d(3,6,5), # > 6 44 44
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 6 22 22
                nn.Conv2d(6,16,5), # 16 18 18

                nn.ReLU(), 
                nn.MaxPool2d(2, 2), # 16 9 9
                nn.Flatten(), # 16*9*9 

                PrintLayer(),
                nn.Linear(before_liner, 256), # 320
                # nn.Linear(157*157*16, 256), # 640
                # nn.Linear(16*9*9, 256), 
                nn.ReLU(),

                nn.Linear(256, num_classes),
    )

    elif model_name == ("res18" or "res18_pre"):
        to_call =  getattr(models, 'resnet18')
        model_ft = to_call(pretrained=use_pretrained)


        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == ("res34" or "res34_pre"):
        to_call =  getattr(models, 'resnet34')
        model_ft = to_call(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == ("res152" or "res152_pre" or "resnet152"):
        to_call =  getattr(models, 'resnet152')
        model_ft = to_call(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']:
        to_call = getattr(models, model_name)
        model_ft = to_call(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)


    elif model_name in [ 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',]:
        torch.use_deterministic_algorithms(False)
        print("Warning: torch.use_deterministic_algorithms set to FALSE")
        
        to_call = getattr(models, model_name)
        model_ft = to_call(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)



    elif model_name in ["alexnet"]:
        """ Alexnet
        """
        to_call = getattr(models, model_name)
        model_ft = to_call(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name in ['squeezenet1_0', 'squeezenet1_1']:
        """ Squeezenet
        """
        to_call = getattr(models, model_name)
        model_ft = to_call(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:
        """ Densenet
        """
        to_call = getattr(models, model_name)
        model_ft = to_call(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name in ['inception_v3']:
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        to_call = getattr(models, model_name)
        model_ft = to_call(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299


    # elif model_name in ['googlenet']:
    #     to_call = getattr(models, model_name)
    #     model_ft = to_call(pretrained=use_pretrained)
    #     set_parameter_requires_grad(model_ft, feature_extract)
    #     num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Linear(num_ftrs, num_classes)


    elif model_name in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',]:
        # to_call = getattr(models, model_name)
        # model_ft = to_call(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)

        from efficientnet_pytorch import EfficientNet
        model_ft = EfficientNet.from_pretrained(model_name)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, num_classes)


    # elif model_name == "":
    #     model_ft = 

    else:
        print(f'>{model_name}<  model Invalid, check models.py, exiting...'  )
        exit()

    return model_ft




