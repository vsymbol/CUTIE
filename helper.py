import requests
import argparse
import sys
import os, re
import time

if __name__ == "__main__":
    src = '/Users/xiaohui.zhao/workspace/CUTIE/hotel_imgs/'
    dst = '/Users/xiaohui.zhao/workspace/data/CUTIE/mixed_true_even/train_hotel'
    json_files = {}
    for dirpath,dirnames,filenames in os.walk(dst):
        for filename in filenames:
            file = re.split(r'\.', filename)[0]
            file_path = os.path.join(dirpath,filename)
            json_files.update({file: file_path})    
    
    files = []
    for dirpath,dirnames,filenames in os.walk(src):
        for filename in filenames:
            #file = os.path.join(dirpath,filename)
            file = re.split(r'\.', filename)[0]
            if file in json_files:
                os.system('cp '+ os.path.join(dirpath,filename) + ' ' + dst)