# general functions that often use  

import os
import pickle
import json
import csv
from time import localtime, strftime
from pathlib import Path
import time
import glob
import re 
import math
from decimal import Decimal
from pathlib import Path
import glob
import re
import torch  # no need repro cause only save here
import os
from tqdm import tqdm
import shutil

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()
 
try:
    import wandb
except ImportError:
    wandb = None








def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'optimizer_state_dict', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1024**2  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path




 





 

def get_stratified(inputs,ratio): # list, float  
    magnitude = math.floor(math.log(len(inputs), 10))  # Ex: 10 > 1, 100 > 2
    margin_ratio = str(round(float(ratio), magnitude))
    numerator, denominator = Decimal(margin_ratio).as_integer_ratio()
    # print(numerator,denominator)
    return [v for i,v in enumerate(inputs) if i % denominator < numerator]




def nowtime(style = 0):
  
  if style == 0:  
    fmt = "%Y%m%d_%H%M%S" 
  elif style == 1:
    fmt = "%Y-%m-%d %H:%M:%S" 
  return strftime(fmt, localtime())

def mkdir(fp):
  Path(fp).mkdir(parents=True, exist_ok=True)

def clean_dir(dir_name ):
    fp_list = glob.glob(  os.path.join(dir_name,'*')) + glob.glob(os.path.join(dir_name,'.*')) 
    for f in tqdm( fp_list, desc=f"Cleaning {dir_name}"  ) :
        # os.remove(f)
        shutil.rmtree(f)


def get_fp_list(dir_name,ext = None):
  fp_list =[]
  for root, dirs, files in os.walk(dir_name):
    for file in files:
      if ext:
        if file.endswith(ext):
          filepath = os.path.join(root, file)
          fp_list += [filepath]
      else:
        filepath = os.path.join(root, file)
        fp_list += [filepath]
  return fp_list



# https://stackoverflow.com/questions/3086973/how-do-i-convert-this-list-of-dictionaries-to-a-csv-file
def dict2csvfile(toCSV,filename = 'tmp.csv',bom = 0,silent=0):
    keys = toCSV[0].keys()
    with open(filename, 'w', encoding='utf-8', newline='')  as output_file:
        if bom: output_file.write('\ufeff')
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)
    if not silent: print('dict2csvfile ok! please check ' + filename)


# https://stackoverflow.com/questions/18337407/saving-utf-8-texts-with-json-dumps-as-utf8-not-as-u-escape-sequence
def dict2jsonfile(dict_src,filename='tmp.json',silent=0):
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(dict_src, fp,indent=4, sort_keys=False,ensure_ascii=False)
    if not silent: print('dict2jsonfile ok! please check '+filename)



def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))






def ins(v):
    print("ins>>>")
    print('>dir:')
    print(dir(v))
    print('>type:')
    print(type(v))
    print('>print:')
    print(v)
    print("ins<<<")


def save_obj(obj1, fp='tmp_obj.pkl',silent = 0):
    if not silent: print('saving obj to ' + fp)
    with open(fp, 'wb') as handle:
        pickle.dump(obj1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if not silent: print('save_obj ok!   '  )
    pass


def load_obj(filename='tmp_obj.txt', silent = 0):
    if not silent: print('loading obj ' + filename)
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    if not silent: print('load_obj ok!  '  )
    return b
    pass





if __name__ == '__main__':
  # do nothing
  print('you called main, do nothing')





