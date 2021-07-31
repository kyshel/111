
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

import os 
import numpy as np
 
# repro 
if os.environ['ICH_REPRO'] == '1':
    seed = 0
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)

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
        x = self.pool(F.relu(self.conv1(x)))      
        x = self.pool(F.relu(self.conv2(x)))      
        x = torch.flatten(x, 1) # flatten all dimensions except batch        
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)
    
        return x

model_first = Net()

model_first_seq =  nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),


            nn.Flatten(),

            nn.Linear(1296, 120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, 7),
)


model_overfit =  nn.Sequential( # > 3 48 48
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


model_basic =  nn.Sequential( # > 3 48 48
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



model_ft  = torchvision.models.resnet18(pretrained=True)
model_ft.fc = nn.Linear(model_ft.fc.in_features, 7)
model_res18 = model_ft



