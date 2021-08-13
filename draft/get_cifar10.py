# get cifar10 from origin, require running in notebook


# # install
# ! pip install cifar2png

# # make
# ! cifar2png cifar10 cifar10_superclass

# re orgnize
from pathlib import Path
from tqdm import tqdm 
from shutil import copyfile 
import os 
import shutil
from pathlib import Path
import pandas as pd


src_dir = 'cifar10_superclass'   # custom
dst_dir = 'cifar10_png'              # custom

total ={'train': 50000,'test': 10000}
names = ['airplane',  'automobile',  'bird',  'cat',  'deer',  'dog',  'frog',  'horse',  'ship',  'truck']
imagelabel_pairs=[]

for split in 'train','test':
    
    g = Path(f'{src_dir}/{split}').rglob('*.png')
    split_dir = f'{dst_dir}/{split}'

    if os.path.exists(split_dir):shutil.rmtree(split_dir)
    Path(split_dir).mkdir(parents=True, exist_ok= True )

    try: tqdm._instances.clear()
    except Exception: pass

    paths = []
    labels = []
    for i,v in enumerate(tqdm( g,total=total[split] ,desc=f'{split}, build paths' )):
        paths += [str(v)]
        labels += [names.index(v.parent.stem) ]
    

    for i,fp in enumerate(tqdm(paths,desc=f'{split}, copy files')):
        image = f'{split}_{i:05d}'
        imagelabel_pairs+= [[image,labels[i]]]

        dst_fp = os.path.join(split_dir,f'{image}.png')
        shutil.copy2(fp,dst_fp) # cp

# make pid2cid map from labels.csv
images,labels = [pair[0] for pair in imagelabel_pairs], [ pair[1] for pair in imagelabel_pairs]
df = pd.DataFrame({'image': images,'label': labels})


csv_fp =os.path.join(dst_dir,'labels.csv')
df.to_csv(csv_fp,index = False)
print(f'Done! check {csv_fp}')