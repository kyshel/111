# ich
![smoke_test](https://github.com/kyshel/ich/actions/workflows/smoke.yml/badge.svg)
A hammer used for image classification tasks. 

## install
``` bash
git clone https://github.com/kyshel/ich.git
cd ich
pip install -r requirements.txt
```

## usage
``` bash
python train.py --data cifar10 --epochs 5 --batch 256 --img-size 32 
```

## test cases
``` bash
python train.py --epochs 3 --batch 256 --img-size 32 --cache
python train.py --epochs 3 --batch 256 --img-size 32
python train.py --epochs 3 --batch 256 --img-size 32 --repro
python train.py --epochs 3 --batch 256 --img-size 32 --kfold 1/5
python train.py --epochs 3 --batch 256 --img-size 32 --skfold 5/5
python train.py --epochs 3 --batch 256 --img-size 16 --skfold 5/5
python train.py --epochs 3 --batch 256 --img-size 16 --skfold 5/5 --workers 2
python train.py --task test 

python train.py --epochs 3 --batch 256 --img-size 16 --skfold 5/5 --name expexp
python train.py --epochs 3 --batch 256 --img-size 16 --skfold 5/5 --name expexp --exist-ok
python train.py --epochs 3 --batch 256 --img-size 16 --skfold 5/5 --name expexp --exist-ok --freeze


```


## p1
- custom dataset add cache 
- add auto download dataset 
- make smoke test pass

## p9
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
- open in colab
- open in kaggle
- docker pull


## logs
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


 
