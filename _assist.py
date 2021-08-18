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
from utils import ax
reload(ax)
import shutil
from tqdm import tqdm 
import multiprocessing.dummy as mp 
from itertools import repeat
import logging
logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)

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

def mesh(pix_input,size = None,surface = None ):
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





def dcm2pix_v2(dcm_raw_pix):
    # has bug, max may == 0
    lut = apply_voi_lut(dcm_raw_pix, dicom)   
    uni = np.amax(lut) - lut  if dicom.PhotometricInterpretation == "MONOCHROME1" else lut

    raw_cut = raw - np.min(raw)
    uni_cut = uni - np.min(uni)


    raw_dot = raw_cut / np.max(raw_cut)
    uni_dot = uni_cut / np.max(uni_cut)


    return raw,raw_cut,uni_cut,raw_dot,uni_dot


def dcm2pix(fp):
    # has bug, max may == 0
    dicom = pydicom.read_file(fp)
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

def seepix(pix,do_norm = False, is_norm = False): 
    if do_norm: pix = (norm(pix) * 255).astype(np.uint8)
    if is_norm: pix = (pix* 255).astype(np.uint8)

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

 




from tqdm import tqdm

SEQS = ['FLAIR' , 'T1w' , 'T1wCE' , 'T2w']
HALT = False
 
def read_pie(pie_dir, pbar = None):
    if 'HALT' in vars() or 'HALT' in globals(): # stop multi-threads
        if HALT: return 

    dcm_files = sorted(
                glob.glob(os.path.join(pie_dir,"*.dcm")), 
                key=lambda x: int(x[:-4].split("-")[-1]),
            )

    pie = []
    for fp in dcm_files:
        dcm = pydicom.read_file(fp)
        pie += [[fp, pydicom.read_file(fp) ]]  # [fp,dcm], ...
        # pie +=  fp, pydicom.read_file(fp)    # [fp,dcm], ...

    if pbar:
        pbar.update(1)

    return pie 

# step1 make dcms 
def read_pies(input_dir,cache_dir= '_pies_cache', workers=8):
    
    # pie dirs
    pie_dirs = {seq:[] for seq in SEQS} # seq: [ pie_dir, pie_dir  ] a pie has many dcms 
    for split in 'train','test':
        case_dirs = [ f.path for f in os.scandir(os.path.join(input_dir,split)) if f.is_dir() ]
        for case_dir in case_dirs:
            for seq in SEQS: #  [inputdir/split/case/seq/] 00000.dcm
                pie_dirs[seq] += [os.path.join(case_dir,seq)]

    # pies
    pies = {seq:[] for seq in SEQS} # { 'FLAIR':[[dcm_fp,dcm_pix],...], ... }
    for seq in SEQS:
        pbar = tqdm(total=len(pie_dirs[seq]), position=0, leave=True,
                    desc = f"read_pies {seq}:", )

        global HALT 
        HALT = False
        try:
            p=mp.Pool(workers)
            pies[seq] = p.starmap(read_pie, zip(pie_dirs[seq], repeat(pbar)))
            p.close()
            pbar.close()
            p.join()
        
            ax.mkdir(cache_dir)
            ax.save_obj(pies[seq],os.path.join(cache_dir, f'{seq}.pkl'))
            pies[seq] = [] # release mem
        except KeyboardInterrupt:
            HALT = True
            raise

    print(f'read_pies all done! check f{cache_dir}')


def make_nobs(input_dir,cache_dir,dst_dir,workers=16,chunksize = 1):  
    # make nobs to write pngs

    # mkdir and prepare labels.csv
    for seq in SEQS:
        for split in 'train','test':
            ax.mkdir(os.path.join(dst_dir,seq,split))
        csv_src = os.path.join(input_dir,'train_labels.csv')
        csv_dst = os.path.join(dst_dir,seq,'labels.csv')
        shutil.copy2(csv_src,csv_dst) # cp

    # make nobs
    for seq in SEQS:
        nobs = []
        pies = ax.load_obj( os.path.join(cache_dir, f'{seq}.pkl') )  # 60s
        # global pies

        gb=0
        results = ThreadPool(workers).imap( make_nob,  pies ,chunksize= chunksize )  
        pbar = tqdm(enumerate(results), total= len(pies)  )
        for i, nob in pbar:
            nobs += [nob]
            gb +=  nob[0].nbytes
            pbar.desc = f'{seq} making nobs ({gb / 1E9:.1f}GB)'
            del nob
        pbar.close()

        ax.save_obj(nobs,os.path.join(cache_dir,f'nobs_{seq}.pkl'))
        # write nobs
        for pix,dst_fp in tqdm(nobs,desc=f'writing nobs to {dst_dir}'):
            pix = kop2rgb2pix(pix,[0.90,0.95]) 
            pix2file(pix,dst_fp,do_norm=1)

        del pies, nobs, pbar, results

    print(f'All Done! Check {dst_dir}')


def pix2im(pix,do_norm = False,is_norm = False,is_expand = False):
    if do_norm and is_norm: raise Exception('do_norm and is_norm can not co-exist')
    if do_norm: pix = (norm(pix) * 255).astype(np.uint8)
    if is_norm: pix = (pix* 255).astype(np.uint8)
 
    im = Image.fromarray(pix) # np > im
    return im


#%% lab, read bad imgs 
def make_nob(pie,workers=16,chunksize = 3):
    # make nob
    pix = 0
    for fp,dcm in pie:  # .dcm  
        raw = dcm.pixel_array
        lut = apply_voi_lut(raw, dcm)   
        uni = np.amax(lut) - lut  if dcm.PhotometricInterpretation == "MONOCHROME1" else lut

        raw_cut = raw - np.min(raw)
        uni_cut = uni - np.min(uni)

        raw_dot = raw_cut / np.max(raw_cut) if  np.max(raw_cut) != 0 else raw_cut + 0.0
        uni_dot = uni_cut / np.max(uni_cut) if  np.max(uni_cut) != 0 else uni_cut + 0.0

        # >>>>>>>>>>>>>>>>>>>>>   core <<<<<<<<<<<<<<<<<<<<<
        # final = dot2pix(kop2rgb(uni_cut,[0.4,0.6], ))
        final = uni_dot

        # print(final.max() )
        pix += final  #  core

    fp = pie[0][0] # should be pie[0][0] after fix 
    segs = fp.split(os.sep)  # [...,'01raw', 'd', -4'train', -3'00466', -2'FLAIR', -1'0.dcm']
    split,fn,seq = segs[-4], f'{segs[-3]}.png', segs[-2]
    dst_fp = os.path.join(dst_dir,seq,split,fn)

    return pix,dst_fp

def scale(x, horizon = None, peak = None):   
    m = horizon if horizon is not None else x.min()  # horizon maybe equal to 0
    n = peak if peak is not None else x.max()
    return (x-m)/(n-m)

def dot2pix(x):
    return ( x * 255.999).astype(np.uint8)

def kop2pix(x):
    return dot2pix(scale(x))

def kop2rgb_v1(kop,seg=[0.6,0.9], horizon = None, peak = None ): 
    # Warning: no rgba include
    # kop = c1to3(kop) if kop.ndim == 2 else (kop )

    dot = scale(kop, horizon = None, peak = None)
    m,n = seg #  1,R,n,G,m,B,0    [0.9,1] [0.6,0.9] [0.3,0.6]
 
    r = np.where( np.logical_and(n < dot , dot <= 1 ), dot, n)
    r = scale(r,n,1)

    g = np.where( np.logical_and(m < dot , dot <= n ), dot , m)
    g = scale(g,m,n)

    b = np.where( np.logical_and(0 <= dot , dot <= m ), dot, 0)
    b = scale(b,0,m)

    return np.dstack((r,g,b))


def kop2rgb(kop,seg=[0.6,0.9], horizon = None, peak = None ): 
    # v2, 0.9 0.95  

    dot = scale(kop, horizon = None, peak = None)
    m,n = seg #  1,R,n,G,m,B,0    [0.9,1] [0.6,0.9] [0.3,0.6]
 
    r = np.where( np.logical_and(n < dot , dot <= 1 ), dot, n)
    r = scale(r,n,1)

    g = np.where( np.logical_and(m < dot , dot <= n ), dot , m)
    g = scale(g,m,n)

    b = dot

    return np.dstack((r,g,b))



def kop2rgb2pix(x,seg=[0.6,0.9]):
    return dot2pix(kop2rgb(x,seg))



dst_dir = '/ap/27bra/03png_pie2'
yes = ['00000','00002','00005','00006','00008','00011','00012','00014','00020','00025','00026','00028','00031','00033','00035','00043','00046','00048','00052','00054','00056','00058','00059','00060','00062','00063','00066','00068','00070','00071','00074','00077','00078','00085','00087','00089','00094','00096','00098','00100','00105','00106','00107','00109','00117','00120','00128','00134','00136','00138','00139','00140','00143','00144','00146','00155','00156','00159','00160','00166','00171','00177','00178','00185','00186','00187','00188','00196','00197','00199','00203','00204','00210','00212','00220','00222','00230','00233','00234','00235','00240','00245','00246','00250','00253','00254','00260','00263','00270','00271','00273','00281','00282','00284','00285','00291','00293','00294','00296','00299','00303','00304','00305','00306','00311','00313','00317','00321','00322','00328','00329','00331','00332','00334','00338','00340','00344','00350','00352','00359','00360','00364','00366','00367','00369','00370','00371','00383','00386','00400','00403','00404','00406','00408','00409','00413','00416','00425','00426','00429','00431','00436','00440','00442','00443','00449','00451','00456','00457','00466','00468','00470','00472','00478','00479','00480','00483','00485','00488','00491','00493','00494','00499','00500','00501','00502','00504','00505','00506','00511','00513','00516','00517','00520','00523','00524','00525','00526','00528','00529','00532','00537','00539','00542','00543','00544','00548','00549','00550','00551','00552','00554','00556','00557','00558','00559','00561','00564','00570','00576','00577','00579','00582','00583','00584','00586','00590','00593','00594','00597','00598','00599','00602','00604','00606','00607','00608','00610','00611','00612','00613','00615','00618','00621','00622','00625','00626','00628','00631','00638','00639','00640','00646','00650','00652','00655','00656','00658','00659','00661','00674','00675','00676','00677','00679','00680','00690','00691','00692','00693','00694','00697','00698','00704','00705','00707','00708','00714','00715','00716','00718','00725','00731','00732','00736','00737','00739','00740','00746','00750','00757','00758','00760','00765','00768','00772','00773','00775','00777','00781','00782','00784','00787','00789','00791','00793','00794','00795','00801','00807','00808','00811','00816','00819','00823','00828','00838','00840','00998','00999','01000','01001','01002','01003','01005','01007','01008',]



no = ['00003','00009','00017','00018','00019','00021','00022','00024','00030','00032','00036','00044','00045','00049','00053','00061','00064','00072','00081','00084','00088','00090','00095','00097','00099','00102','00104','00108','00110','00111','00112','00113','00116','00121','00122','00123','00124','00130','00132','00133','00137','00142','00147','00148','00149','00150','00151','00154','00157','00158','00162','00165','00167','00169','00170','00172','00176','00183','00184','00191','00192','00193','00194','00195','00201','00206','00209','00211','00214','00216','00217','00218','00219','00221','00227','00228','00231','00236','00237','00238','00239','00241','00242','00243','00247','00249','00251','00258','00259','00261','00262','00266','00267','00269','00274','00275','00280','00283','00286','00288','00289','00290','00297','00298','00300','00301','00308','00309','00310','00312','00314','00316','00318','00320','00324','00325','00327','00336','00339','00341','00343','00346','00347','00348','00349','00351','00353','00356','00373','00376','00377','00378','00379','00380','00382','00387','00388','00389','00390','00391','00392','00395','00397','00399','00401','00402','00405','00407','00410','00412','00414','00417','00418','00419','00421','00423','00430','00432','00433','00441','00444','00445','00446','00452','00454','00455','00459','00464','00469','00477','00481','00495','00496','00498','00507','00510','00512','00514','00518','00519','00530','00533','00538','00540','00545','00547','00555','00563','00565','00567','00568','00569','00571','00572','00574','00575','00578','00581','00587','00588','00589','00591','00596','00601','00605','00616','00619','00620','00623','00624','00630','00636','00641','00642','00645','00649','00651','00654','00657','00663','00667','00668','00682','00683','00684','00685','00686','00687','00688','00703','00706','00709','00723','00724','00727','00728','00729','00730','00733','00734','00735','00742','00744','00747','00751','00753','00756','00759','00764','00767','00774','00778','00780','00788','00792','00796','00797','00799','00800','00802','00803','00804','00805','00806','00809','00810','00814','00818','00820','00824','00830','00834','00836','00837','00839','01004','01009','01010',]



SEQS = ['FLAIR' , 'T1w' , 'T1wCE' , 'T2w']
for seq in SEQS:
    # multi dcms squash to nob
    pie = read_pie(f'/ap/27bra/01raw/d/train/{yes[0]}/{seq}')
    # print(pie[0][1].pixel_array)
    pix,dst_fp = make_nob(pie)
    # pix = kop2rgb2pix(pix,[0.90,0.95]) 
    im = pix2im(pix ,do_norm = 1)
    im.show()

print('ok!')


# # single dcm 
# raw,raw_cut,uni_cut,raw_dot,uni_dot = \
#     dcm2pix(f'/ap/27bra/01raw/d/train/00000/FLAIR/Image-250.dcm')

# pix = kop2pix(raw)
# # pix = kop2rgb2pix(uni_cut,[0.3,0.7])
# im = pix2im(pix)
# im.show()

print(111)


# %% make_pie v2

input_dir = '/ap/27bra/01raw/d/'
cache_dir = '_pies_cache'
dst_dir = '/ap/27bra/03png_pie2'
# read_pies(input_dir,cache_dir, workers=16) # ONCE!
make_nobs(input_dir,cache_dir,dst_dir,workers=16,chunksize=3)



#%% stale, origin version, make pies

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
        

    # make pies
    pbar = tqdm(total=len(dcms_dirs), position=0, leave=True,
              desc = f"dcms2pie, {input_dir} > {dst_dir}: ",
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    p=mp.Pool(workers)
    p.starmap(dcms2pie, zip(dcms_dirs, dst_files,repeat(pbar)))
    p.close()
    pbar.close()
    p.join()


    # # v2 version 
    # # read dcms and save to a list 
    # for c in cohorts: # prevent mem boom 
    #     dcms_list = 
    #     pass
    # # write list to dst_fp
    # for c in cohorts:
    #     pass


# make_pies(input_dir,dst_dir,workers=16)


input_dir = '/ap/27bra/01raw/d/'
dst_dir = '/ap/27bra/03png_pie2'
# make_pies(input_dir,dst_dir,workers=16)





HALT = False
# ax.clean_dir(dst_dir)
try:
    make_pies(input_dir,dst_dir,workers=8)
except KeyboardInterrupt:
    HALT = True
    raise





#%% lab: stale multi thread slice 

################ multi thread ######################
# results = ThreadPool(workers).imap( make_slice, zip(pie[0::2],pie[1::2]),chunksize=chunksize)  
# for uni_cut in results:
#     pix += uni_cut


# def make_slice(dcm_info): # not inc peformance
#     _fp,dcm = dcm_info
#     raw = dcm.pixel_array
#     lut = apply_voi_lut(raw, dcm)   
#     uni = np.amax(lut) - lut  if dcm.PhotometricInterpretation == "MONOCHROME1" else lut

#     # raw_cut = raw - np.min(raw)
#     uni_cut = uni - np.min(uni)

#     # raw_dot = raw_cut / np.max(raw_cut) if  np.max(raw_cut) != 0 else raw_cut + 0
#     # uni_dot = uni_cut / np.max(uni_cut) if  np.max(uni_cut) != 0 else uni_cut + 0
#     return uni_cut


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



# %% lab, lena  pix2file
import png
def pix2file(pix,dst_fp,do_norm = False,is_norm = False,expand = False):
 

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







