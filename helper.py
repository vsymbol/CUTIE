import requests
import argparse
import sys
import os, re, json
import time

# if __name__ == "__main__":
#     src = '/Users/xiaohui.zhao/workspace/CUTIE/hotel_imgs/'
#     dst = '/Users/xiaohui.zhao/workspace/data/CUTIE/mixed_true_even/train_hotel'
#     json_files = {}
#     for dirpath,dirnames,filenames in os.walk(dst):
#         for filename in filenames:
#             file = re.split(r'\.', filename)[0]
#             file_path = os.path.join(dirpath,filename)
#             json_files.update({file: file_path})    
#     
#     files = []
#     for dirpath,dirnames,filenames in os.walk(src):
#         for filename in filenames:
#             #file = os.path.join(dirpath,filename)
#             file = re.split(r'\.', filename)[0]
#             if file in json_files:
#                 os.system('cp '+ os.path.join(dirpath,filename) + ' ' + dst)

if __name__ == "__main__":
    src = '/Users/xiaohui.zhao/workspace/data/CUTIE/column_identity'
    dst = '/Users/xiaohui.zhao/workspace/data/CUTIE/column'
    json_files = {}
    for dirpath,dirnames,filenames in os.walk(src):
        for filename in filenames:
            file_path = os.path.join(dirpath,filename)      
            if file_path[-3:] == 'png':
                continue    
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)
                print(len(data['fields']))
                for i in range(len(data['fields'])):
                    data['fields'][i]['field_name'] = 'Column{}'.format(i)
                    
            target_path = os.path.join(dst,filename)     
            with open(target_path, 'w') as f:
                json.dump(data, f)
    
    