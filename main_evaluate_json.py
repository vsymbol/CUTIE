# -*- coding: utf-8 -*-
# written by Xiaohui Zhao
# 2018-01 
# xiaohui.zhao@accenture.com
import tensorflow as tf
import numpy as np
import argparse
import os, csv
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model_cutie import CUTIE
from model_cutie_res import CUTIERes
from model_cutie_unet8 import CUTIEUNet
from model_cutie_sep import CUTIESep
from data_loader_json import DataLoader
from utils import *

parser = argparse.ArgumentParser(description='CUTIE parameters')
parser.add_argument('--doc_path', type=str, default='data/meals_1215/') # modify this
parser.add_argument('--save_prefix', type=str, default='meals', help='prefix for ckpt and results') # modify this

parser.add_argument('--fill_bbox', type=bool, default=False) # augment data row/col in each batch

parser.add_argument('--e_ckpt_path', type=str, default='../graph/CUTIE/graph/') # modify this
parser.add_argument('--ckpt_file', type=str, default='CUTIE_residual_8x_d29939c9(r80c80)_iter_45401.ckpt')
parser.add_argument('--rows_ulimit', type=int, default=80) 
parser.add_argument('--cols_ulimit', type=int, default=80) 
parser.add_argument('--rows_blimit', type=int, default=64) 
parser.add_argument('--cols_blimit', type=int, default=64) 

parser.add_argument('--load_dict', type=bool, default=True, help='True to work based on an existing dict') 
parser.add_argument('--load_dict_from_path', type=str, default='dict/40000') # 40000 or 119547 or 20000TC
parser.add_argument('--tokenize', type=bool, default=True) # tokenize input text
parser.add_argument('--text_case', type=bool, default=True) # case sensitive
parser.add_argument('--dict_path', type=str, default='dict/---') # not used if load_dict is True

parser.add_argument('--restore_ckpt', type=bool, default=True) 

parser.add_argument('--embedding_size', type=int, default=128) 
parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--c_threshold', type=float, default=0.5) 
params = parser.parse_args()

if __name__ == '__main__':
    # data
    #data_loader = DataLoader(params, True, True) # True to use 25% training data
    data_loader = DataLoader(params, update_dict=False, load_dictionary=True, data_split=0) # False to provide a path with only test data
    num_words = max(20000, data_loader.num_words)
    num_classes = data_loader.num_classes

    # model
    network = CUTIERes(num_words, num_classes, params, False)   
    model_output = network.get_output('softmax')
    
    # evaluation
    ckpt_saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        try:
            ckpt_path = os.path.join(params.e_ckpt_path, params.save_prefix, params.ckpt_file)
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            print('Restoring from {}...'.format(ckpt_path))
            ckpt_saver.restore(sess, ckpt_path)
            print('{} restored'.format(ckpt_path))
        except:
            raise Exception('Check your pretrained {:s}'.format(ckpt_path))
            
        # calculate validation accuracy and display results   
        recalls, accs_strict, accs_soft = [], [], []
        num_test = len(data_loader.validation_docs)
        for i in range(num_test):
            data = data_loader.fetch_validation_data()
            print('{:d} samples left to be tested'.format(num_test-i))
            
            grid_table = data['grid_table']
            gt_classes = data['gt_classes']
            feed_dict = {
                network.data: grid_table
            }
            fetches = [model_output]
            
            [model_output_val] = sess.run(fetches=fetches, feed_dict=feed_dict)
            recall, acc_strict, acc_soft, res = cal_accuracy(data_loader, np.array(data['grid_table']), 
                                                   np.array(data['gt_classes']), model_output_val, 
                                                   np.array(data['label_mapids']), data['bbox_mapids'])                  
#             recall, acc_strict, res = cal_save_results(data_loader, params.save_prefix,
#                                                        np.array(docs), 
#                                                        np.array(grid_table), 
#                                                        np.array(gt_classes), 
#                                                        model_output_val)
            recalls += [recall]
            accs_strict += [acc_strict] 
            accs_soft += [acc_soft]
            #print(res) # show res for current batch  
            
            # visualize result
            shape = data['shape']
            file_name = data['file_name'][0] # use one single file_name
            bboxes = data['bboxes'][file_name]
            vis_bbox(data_loader, params.doc_path, np.array(grid_table)[0], 
                     np.array(gt_classes)[0], np.array(model_output_val)[0], file_name, 
                     np.array(bboxes), shape)

        recall = sum(recalls) / len(recalls)
        acc_strict = sum(accs_strict) / len(accs_strict)
        acc_soft = sum(accs_soft) / len(accs_soft)
        print('EVALUATION ACC (Recall/Acc): %.3f / %.3f (%.3f) \n'%(recall, acc_strict, acc_soft))