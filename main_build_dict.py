# written by Xiaohui Zhao
# 2018-01
# xiaohui.zhao@outlook.com
import tensorflow as tf
import argparse

from data_loader_json import DataLoader

parser = argparse.ArgumentParser(description='CUTIE parameters')
parser.add_argument('--dict_path', type=str, default='dict/') 
parser.add_argument('--doc_path', type=str, default='invoice_data/') 
parser.add_argument('--test_path', type=str, default='') # leave empty if no test data provided
parser.add_argument('--text_case', type=bool, default=True) # case sensitive
parser.add_argument('--tokenize', type=bool, default=True) # tokenize input text
parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--use_cutie2', type=bool, default=False) 
params = parser.parse_args()

if __name__ == '__main__':
    ## run this program before training to create a basic dictionary for training
    data_loader = DataLoader(params, update_dict=True, load_dictionary=False)
