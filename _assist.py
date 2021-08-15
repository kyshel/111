# snippets that assist devleop
# type: ignore
# flake8: noqa

#%% func  444

%matplotlib inline
import pydicom  
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np 
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import ipyvolume as ipv
import numpy as np
import matplotlib
from importlib import reload
import glob 
import os 
from multiprocessing.pool import ThreadPool

reload(matplotlib)
matplotlib.style.use('dark_background')

def c1to3(a):
    return  np.repeat(np.expand_dims(a, axis=2),3,axis = 2)

def c3to1(pix):
    return pix[:,:,0]

def rot270(a):
    return np.rot90(np.rot90(np.rot90(a)))

def rot180(a):
    return np.rot90(np.rot90(a))

def rot90(a):
    return  np.rot90(a)    

def rot0(a):
    return a

def resize(im_np,size):
    return np.asarray(T.Resize(size=size)(Image.fromarray(im_np)))

def gray_rm(im_np):
    return np.asarray(T.Grayscale()(Image.fromarray(im_np)))

def gray_rm(im_np):
    return np.asarray( Image.fromarray(im_np).convert('L')  )

def gray(im_np,is_round = False):
    if im_np.ndim == 2:
        return  im_np
    else:
        im2 =np.zeros([im_np.shape[0],im_np.shape[1]])
        for i,r in enumerate(im_np):
            for j,v in enumerate(r):
                # R * 299/1000 + G * 587/1000 + B * 114/1000
                im2[i][j] = v[0]* 299/1000 + v[1]* 587/1000 + v[2]* 114/1000 
                if   is_round :  im2[i][j]  = round(im2[i][j]  )
        return  im2

def hist(input_2d,bins = 'auto'):
    y, x = np.histogram(input_2d.ravel(), bins=bins)
    fig, ax = plt.subplots()
    ax.plot(x[:-1], y)
    fig.show()
    plt.show()

def mesh(pix_input,size = None):
    # show mesh, pix gray must 1 channel
    if size is None: size = pix_input.shape[0] , pix_input.shape[1]

    # print(pix_input.max())
    pix = resize(pix_input,size)
    # print(pix.max())  # will decay

    a = np.arange(0, pix.shape[0])
    X, Y = np.meshgrid(a, a)
    Z = rot270(gray(pix))   # surface, 1 channel
    # Z = rot270( pix )

    deno = pix.max() - pix.min()
    c = rot270(c1to3(pix) if pix.ndim == 2 else pix)/deno     # color , 3 channel 


    ipv.figure()
    
    # mesh = ipv.plot_mesh(Y,X,Z,    wireframe=True,color=c )
    ipv.plot_wireframe(Y,X,Z, color=c)
    ipv.show()
    ipv.style.set_style_dark()


def dcm2pix(dcm):
    # has bug, max may == 0
    dicom = pydicom.read_file(dcm)
    raw = dicom.pixel_array
    lut = apply_voi_lut(raw, dicom)   
    uni = np.amax(lut) - lut  if dicom.PhotometricInterpretation == "MONOCHROME1" else lut

    raw_cut = raw - np.min(raw)
    uni_cut = uni - np.min(uni)

    # raw_cut_max = 0.000001 if np.max(raw_cut) == 0 else np.max(raw_cut)
    # uni_cut_max = 0.000001 if np.max(raw_cut) == 0 else np.max(raw_cut)

    raw_dot = raw_cut / np.max(raw_cut)
    uni_dot = uni_cut / np.max(uni_cut)

    # print(f'raw: min {raw.min(): <8} max {raw.max(): <8}')
    # print(f'raw_cut: min {raw_cut.min(): <8} max {raw_cut.max(): <8}')
    # print(f'uni_cut: min {uni_cut.min(): <8} max {uni_cut.max(): <8}')
    # print(f'raw_dot: min {raw_dot.min(): <8} max {raw_dot.max(): <8}')
    # print(f'uni_dot: min {uni_dot.min(): <8} max {uni_dot.max(): <8}')

    # hist(raw)
    # hist(raw_cut)
    # hist(uni_cut)
    # hist(raw_dot)
    # hist(uni_dot)

    return raw,raw_cut,uni_cut,raw_dot,uni_dot


def pix2file(pix,dst_fp,do_norm = False,is_norm = False):
    if do_norm and is_norm: raise Exception('do_norm and is_norm can not co-exist')
    if do_norm: pix = (norm(pix) * 255).astype(np.uint8)
    if is_norm: pix = (pix* 255).astype(np.uint8)
 
    im = Image.fromarray(pix) # np > im
    im.save(dst_fp)



 


def seepix(pix,do_norm = False): 
    if do_norm: pix = (norm(pix) * 255).astype(np.uint8)
    im = Image.fromarray(pix) # np > im 
    im.show()

def norm(pix):
    return pix / pix.max()

def dcms2pie(dcms_fp,dst_fp = None,pbar=None): 
    if 'HALT' in vars() or 'HALT' in globals(): # stop multi-threads
        if HALT: return 

    dcm_files = sorted(
                glob.glob(os.path.join(dcms_fp,"*.dcm")), 
                key=lambda x: int(x[:-4].split("-")[-1]),
            )

    raw,raw_cut,uni_cut,raw_dot,uni_dot = dcm2pix(dcm_files[0])
    
    pie = np.zeros(raw.shape)
    for f in dcm_files:
        raw,raw_cut,uni_cut,raw_dot,uni_dot = dcm2pix(f)
        
        pie += raw  # must raw here, 1 pie should has same scale 

    if dst_fp is not None: # make file
        pix2file(pie,dst_fp,do_norm=True) 

    if pbar:
        pbar.update(1)

    return pie


#%% make pies

from utils import ax
reload(ax)
import shutil
from tqdm import tqdm 
import multiprocessing.dummy as mp 
from itertools import repeat

def make_pies(input_dir,dst_dir,labels_csv_name = 'train_labels.csv',workers = 16):
    cohorts = ['FLAIR' , 'T1w' , 'T1wCE' , 'T2w']

    # dcms_dirs
    dcms_dirs = []
    for split in 'train','test':
        case_dirs = [ f.path for f in os.scandir(os.path.join(input_dir,split)) if f.is_dir() ]
        for case_dir in case_dirs:
            for cohort in cohorts:

                pie_dir = os.path.join(case_dir,cohort)
                dcms_dirs += [pie_dir]

    # mkdir and prepare labels.csv
    for c in cohorts:
        for split in 'train','test':
            ax.mkdir(os.path.join(dst_dir,c,split))
        csv_src = os.path.join(input_dir,labels_csv_name)
        csv_dst = os.path.join(dst_dir,c,'labels.csv')
        shutil.copy2(csv_src,csv_dst) # cp
            
    # dst_files
    dcms_dirs = sorted(dcms_dirs)
    dst_files = []
    for p in dcms_dirs:
        segs = p.split(os.sep)  # [...,'01raw', 'd', 'train', '00466', 'FLAIR']
        # print(segs)
        cohort = segs[-1]
        split =  segs[-3]
        fn = f'{segs[-2]}.png'
        dst_fp = os.path.join(dst_dir,cohort,split,fn)
        # print(p,dst_fp)
        dst_files += [dst_fp]
        

    # print(dcms_dirs[2],dst_files[2])
    # dcms2pie(dcms_dirs[2],dst_files[2]) # pie need norm 

    # print(dcms_dirs,dst_files)

    # make pies
    # gb=0
    # results = ThreadPool(workers).imap(dcms2pie,   zip(dcms_dirs, dst_files   )) 
    # pbar = tqdm(enumerate(results) , mininterval=1,)
    # for i, x in pbar:
    #     # gb += self.pixs[i].nbytes # sys.getsizeof(b.storage())
    #     gb +=  x.nbytes
    #     pbar.desc = f' Cooking pies ({gb / 1E9:.1f}GB)'
    # pbar.close()

    # make pies
    pbar = tqdm(total=len(dcms_dirs), position=0, leave=True,
              desc = f"dcms2pie, {input_dir} > {dst_dir}: ",
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    p=mp.Pool(workers)
    p.starmap(dcms2pie, zip(dcms_dirs, dst_files,repeat(pbar)))
    p.close()
    pbar.close()
    p.join()




# make_pies(input_dir,dst_dir,workers=16)


input_dir = '/ap/27bra/01raw/d/'
dst_dir = '/ap/27bra/03png_pie2'
# make_pies(input_dir,dst_dir,workers=16)





HALT = False
# ax.clean_dir(dst_dir)
try:
    make_pies(input_dir,dst_dir,workers=16)
except KeyboardInterrupt:
    HALT = True
    raise



#%% lab: load dicom 
 
fp = '/ap/27bra/01raw/d/train/00000/T1wCE/Image-78.dcm'
fp = '/ap/27bra/01raw/d/train/00000/FLAIR/Image-50.dcm'
raw,raw_cut,uni_cut,raw_dot,uni_dot = dcm2pix(fp)
print(raw)
# w = 64 ; mesh(raw,(w,w))
# dcms2spie
pie = dcms2pie('/ap/27bra/01raw/d/train/00000/FLAIR')
# # w = 64 ; mesh(norm(pix),(w,w))
 
# %% lab: resize will decay max 
row = 4
row2 = 2
a = np.arange(row*row).reshape(row,row).astype(np.uint8)
a_resized  = resize(a,(row2,row2))
# a = np.linspace(0,1,row*row).reshape(row,row)   #.astype(np.uint8)
a_dot = a / np.max(a)
a_dot_resized = resize(a_dot,(row2,row2))
a , a_dot, a_resized, a_dot_resized

# %%
mesh(raw,(128,128))



# %% lena  pix2file
import png
def pix2file(pix,dst_fp,do_norm = False,is_norm = False):
    if do_norm and is_norm: raise Exception('do_norm and is_norm can not co-exist')
    if do_norm: pix = (norm(pix) * 255).astype(np.uint8)
    if is_norm: pix = (pix* 255).astype(np.uint8)
 
    im = Image.fromarray(pix) # np > im
    im.save(dst_fp)


fp = 'lena.png'
im = Image.open(fp) 
im_gray = im.convert('RGB') 
pix_gray = np.array(im_gray) 

pix2file(pix_gray,'lena_rm2.png')


# %% mesh lena 

fp = 'lena.png'
im = Image.open(fp) 
im_gray = im.convert('L') 
pix = np.array(im)                 #  rgb  3c
pix_gray = np.array(im_gray)           # gray 1c
pix_32 = np.array(im.resize([32,32]))    

# width = 32
# mesh(pix_gray,(width,width))
# mesh(pix,(width,width))

mesh(pix,(512,512))
mesh(pix_32)


raise # cut
 






















# %%    3d plot surface     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# preset 
%matplotlib inline
import matplotlib
from importlib import reload
reload(matplotlib)
matplotlib.style.use('dark_background')

def c1to3(a):
    return  np.repeat(np.expand_dims(a, axis=2),3,axis = 2)


# %%
print(1)

#%% im
from PIL import Image
import numpy as np


dim = 128
plot_shape = (dim,dim) # w, h

fp = 'lena.png'
im_raw = Image.open(fp) 
im = im_raw.convert('L').resize(plot_shape)
# im.show()

pix = np.array(im)   # gray
print(pix.shape)

im2 = im.convert('RGB')
pix2 = np.array(im2)  # gray 3c

pix3 = np.array(im_raw.resize(plot_shape))   #  rgb 

#%% plot 
import ipyvolume as ipv
import numpy as np
 
a = np.arange(0, pix.shape[0])
U, V = np.meshgrid(a, a)
X = U
Y = V
Z = pix

ipv.figure()
# ipv.plot_surface(X,Y, Z, color=pix2/255)
# ipv.plot_mesh(X,Y, Z, color=pix3/255, wireframe=True)
ipv.plot_wireframe(X, Y,  Z, color=pix3/255)
ipv.show()

ipv.style.set_style_dark()
ipv.view(azimuth=0, elevation=0, distance=10)


# ipv.style.use('black')
# ipv.style.background_color('black')














#%%
ipv.view(azimuth=180, elevation=0, distance=5)






#%% 1 channel to 3 channel 
# np.repeat(  [[0,1],[1,2]], 3 , axis = np.newaxis )

def c1to3(a):
    return  np.repeat(np.expand_dims(a, axis=2),3,axis = 2)

a = np.array([[1,2],[3,4]])
b = c1to3(a)

a ,   b   ,  a.shape,   b.shape

# %% PIL load image
from PIL import Image
import numpy as np
plot_shape = (128,128) # w, h

fp = 'lena.png'
im = Image.open(fp).convert('L')
im = im.resize(plot_shape)
im.show()

pix = np.array(im)
print(pix.shape)



import ipyvolume as ipv
import numpy as np
 
a = np.arange(0, pix.shape[0]-1)
U, V = np.meshgrid(a, a)
X = U
Y = V
Z = pix


 
ipv.figure()
ipv.plot_surface(X, Z, Y, color="gray")
# ipv.plot_wireframe(X, Z, Y, color="green")
ipv.show()
ipv.style.set_style_dark()





# %% show surface plot

from mpl_toolkits import mplot3d
import numpy as np
 
import matplotlib.pyplot as plt
x = np.outer(np.linspace(0, plot_shape[0]-1, plot_shape[0]), np.ones(plot_shape[0]))
y = x.copy().T # transpose
# z = np.cos(x ** 2 + y ** 2)
z = pix



fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()






# %% set ipv
import ipyvolume as ipv
import numpy as np

# reload(ipv)
# ipv.style.use('dark')

# f(u, v) -> (u, v, u*v**2)
a = np.arange(0, pix.shape[0]-1)
U, V = np.meshgrid(a, a)
X = U
Y = V
Z = pix


 
ipv.figure()
ipv.plot_surface(X, Z, Y, color="orange")
ipv.plot_wireframe(X, Z, Y, color="red")
ipv.show()
ipv.style.set_style_dark()




#%%


s = 1/2**0.5
# 4 vertices for the tetrahedron
x = np.array([1.,  -1, 0,  0])
y = np.array([0,   0, 1., -1])
z = np.array([-s, -s, s,  s])
# and 4 surfaces (triangles), where the number refer to the vertex index
triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1,3,2)]



ipv.figure()
# we draw the tetrahedron
mesh = ipv.plot_trisurf(x, y, z, triangles=triangles, color='orange')
# and also mark the vertices
ipv.scatter(x, y, z, marker='sphere', color='blue')
ipv.xyzlim(-2, 2)
ipv.show()
# %%

# %%





# %% ipywidgets slider  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import ipywidgets as widgets

out = widgets.Output()
def on_value_change(change):
    with out:
        print(change['new'])

slider = widgets.IntSlider(min=1, max=100, step=1, continuous_update=True)
play = widgets.Play(min=1, interval=2000)

slider.observe(on_value_change, 'value')
widgets.jslink((play, 'value'), (slider, 'value'))
widgets.VBox([play, slider, out])


 
# %%  plot3D            >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

X = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

Y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Z = [21384, 29976, 15216, 4584, 10236, 7546, 6564, 2844, 4926, 7722, 4980, 2462, 12768, 9666, 2948, 6548, 10776, 8260, 8674, 5584, 5382, 10542, 12544, 5268, 25888, 31220, 9064, 7536, 6618, 1928, 9030, 5790, 6076, 8290, 8692, 4006, 14722, 11016, 2818, 9458, 3054, 5976, 1102, 1084, 9700, 8904, 12510, 11176, 10712, 6548, 2600, 5070, 6538, 4514, 1036, 292, 12572, 6534, 4478, 18500, 10452, 1912, 14254, 31050, 3880, 744, 990, 5534, 1670, 446, 2778, 8272, 14726, 27094, 872, 418, 884, 476, 2806, 1246, 1140, 922, 6202, 10848, 28828, 2360, 9660, 1412, 4296, 5272, 2854, 4150, 770, 5628, 4676, 3500, 31220, 10480, 5704, 5550, 1528, 3168, 2092, 2056, 1874, 7312, 938, 7428]

x = np.reshape(X, (9, 12))
y = np.reshape(Y, (9, 12))
z = np.reshape(Z, (9, 12))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# generate some sample data
import scipy.misc
lena = cv2.imread("lena.png", 0)

# downscaling has a "smoothing" effect
lena = cv2.resize(lena, (100,100))

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.jet,
                linewidth=0)

# show it
plt.show()







# %%


import numpy as np
import torch
import sys


# dim = [1]

! free -h | awk '{print $3}'  | sed -n 2p

dim = [1,1024,1024,1024]
# n = np.zeros(dim, dtype='float32')          # 4GB = 1024 * 1024 * 1024 * 8B
t = torch.zeros(dim,dtype = torch.float32)  # 4GB = 1024 * 1024 * 1024 * 8B

! free -h | awk '{print $3}'  | sed -n 2p



print(f'1E9:{sys.getsizeof(t.storage()) / 1E9:.1f}, 1024**3:{sys.getsizeof(t.storage()) / 1024**3:.1f}')



# %% download url


import os
os.environ['http_proxy'] = "http://127.0.0.1:1080" 
os.environ['https_proxy'] = "http://127.0.0.1:1080" 

import requests
requests.get("http://google.com")



from torchvision.datasets.utils import check_integrity, download_and_extract_archive


a = download_and_extract_archive(
    url = 'https://github.com/kyshel/cifar10/releases/download/v1/cifar10_png.tgz',
    download_root = 'lab' ,

)

print(a)



# %% 
exit()



a = [1,2]
b,c = a
print(b,c)




exit()

import ax
from pathlib import Path

a = 123
fp  = '_rm6666/_rm5555555/_rm2222.txt'


ax.mkdir(Path(fp).parent)

exit()



# %% calc dataset mean&std
from tqdm import tqdm 
import torch

def get_mean_and_std2(dataset,batch=1):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch )
 
    mean = 0.0
    for images, _ in tqdm(loader):
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*224*224))


  
    print(mean,std)



def get_mean_and_std(dataset,batch=1):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch,)
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(tqdm(loader),0):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)
    print(mean,std)

 
transform = transforms.Compose([transforms.Resize([2000,2000]),
    transforms.ToTensor(),])

from datasets import Covid
d1 = Covid(
        root='../03png/train', 
        obj = '../11data/train_imginfo.obj',
        train=True,
        transform=transform,
        workers = opt.workers,
        )

d2 = Covid(
        root='../03png/test', 
        obj = '../11data/test_imginfo.obj',
        train=False,                    
        transform=transform ,
        workers = opt.workers,
        
        )


get_mean_and_std(d1,batch = opt.batch)
get_mean_and_std(d2,batch = opt.batch)













# %%



import numpy as np


np.random.RandomState(0)







