# written by Xiaohui Zhao
# 2018-01
# xiaohui.zhao@accenture.com
import tensorflow as tf
import argparse

from data_loader_json import DataLoader

parser = argparse.ArgumentParser(description='CUTIE parameters')
parser.add_argument('--dict_path', type=str, default='dict/taxi') 
parser.add_argument('--doc_path', type=str, default='data/taxi') 
parser.add_argument('--save_prefix', type=str, default='taxi', help='prefix for ckpt and dict') # TBD: save log/models with prefix
parser.add_argument('--batch_size', type=int, default=32) 
params = parser.parse_args()

if __name__ == '__main__':
    ## run this program before training to create a basic dictionary for training
    data_loader = DataLoader(params, for_train=True, load_dictionary=False)