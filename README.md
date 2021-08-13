# ich  
[![basic_test](https://github.com/kyshel/ich/actions/workflows/basic.yml/badge.svg)](https://github.com/kyshel/ich/actions/workflows/basic.yml)
[![mass_test](https://github.com/kyshel/ich/actions/workflows/mass.yml/badge.svg)](https://github.com/kyshel/ich/actions/workflows/mass.yml)

A hammer used for image classification tasks. 

## install
``` bash
git clone https://github.com/kyshel/ich.git
cd ich
pip install -r requirements.txt
```

## usage
``` bash
python train.py  # try sample
python train.py --epochs 60 --batch 1024 --img-size 32  # customize 
```
 


## p1
- none


## p9
- pretty progress bar 4row to 2row
- torch.resize  strech or keep ratio ? 
- refactor custom
- open in colab
- open in kaggle
- docker pull


## logs
- v26 add badge 
- v25 add --task
- v24 refactor: rename funcs, like pix,cid,fn
- v23 add dataset img autocheck 
- v22 make smoke test pass
- v21 add auto download dataset 
- v20 custom dataset add cache 
- v19 add custom dataset
- v18 remove dev.yaml loose load, add strict mode
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

## leave it 
- repro not work in notebook
- isinteractive not reliable

## devlop info
Developed in vscode with interactive features


 
