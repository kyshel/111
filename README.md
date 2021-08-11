# ich
A hammer used for image classification tasks. 

# install
``` bash
git clone https://github.com/kyshel/ich.git
cd ich
pip install -r requirements.txt
```

# usage
``` bash
python train.py 
python train.py --model resnet18 
```

# p1
- add custom dataset
- remove dev.yaml loose load, add strict mode

# p9
- add badge 
- add download.py 
- add actions for smoke
- add --task
- unify dataset
- fit custom dataset to official dataset 
- refactor: rename funcs, like pix,cid,fn
- pretty progress bar 4row to 2row
- add dataset img autocheck 
- torch.resize  strech or keep ratio ? 
- refactor custom


# logs
- v17 add transformed dataset cache 
- v16 add kfold and skolf  
- v15 save .pt with optimzer not striped
- v14 add model selection
- v13 logger to local file ok 
- v12 reproducibility ok
- v11 tensorboard ok
- v10 wandb resume ok
- v9 wandb ok
- v8 resume ok
- v7 add checkpoint 
- v6 save last.pt and best.pt

# leave it 
- repro not work in notebook
- isinteractive not reliable

# devlop info
Developed in vscode with interactive features


 
