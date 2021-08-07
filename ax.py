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


os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()


 

 





def nowtime(style = 0):
  
  if style == 0:
    fmt = "%Y%m%d_%H%M%S" 
  elif style == 1:
    fmt = "%Y-%m-%d %H:%M:%S" 
  return strftime(fmt, localtime())

def mkdir(fp):
  Path(fp).mkdir(parents=True, exist_ok=True)


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


def save_obj(obj1, fp='tmp_obj.txt',silent = 0):
    with open(fp, 'wb') as handle:
        pickle.dump(obj1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if not silent: print('save_obj ok! please check ' + fp)
    pass


def load_obj(filename='tmp_obj.txt', silent = 0):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    if not silent: print('load_obj ok! src_file: ' + filename)
    return b
    pass





if __name__ == '__main__':
  # do nothing
  print('you called main, do nothing')





