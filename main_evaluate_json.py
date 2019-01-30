# -*- coding: utf-8 -*-
# written by Xiaohui Zhao
# 2018-01 
# xiaohui.zhao@accenture.com
import tensorflow as tf
import numpy as np
import argparse
import os, csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model_cutie import CUTIE
from model_cutie_res import CUTIERes
from model_cutie_unet8 import CUTIEUNet
from model_cutie_sep import CUTIESep
from data_loader_json import DataLoader

parser = argparse.ArgumentParser(description='CUTIE parameters')
parser.add_argument('--doc_path', type=str, default='data/meals_1215') # modify this
parser.add_argument('--save_prefix', type=str, default='meals', help='prefix for ckpt and results') # modify this

parser.add_argument('--e_ckpt_path', type=str, default='../graph/CUTIE/graph/') # modify this
parser.add_argument('--ckpt_file', type=str, default='CUTIE_residual_8x_40000x9_iter_10000.ckpt')  

parser.add_argument('--load_dict', type=bool, default=True, help='True to work based on an existing dict') 
parser.add_argument('--load_dict_from_path', type=str, default='dict/40000') # 40000 or 119547  
parser.add_argument('--dict_path', type=str, default='dict/---') # not used if load_dict is True

parser.add_argument('--restore_ckpt', type=bool, default=True) 

parser.add_argument('--embedding_size', type=int, default=120) 
parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--c_threshold', type=float, default=0.5) 
params = parser.parse_args()

def cal_save_results(docs, grid_table, gt_classes, model_output_val):
    res = ''
    num_correct = 0
    num_correct_strict = 0
    num_all = grid_table.shape[0] * (model_output_val.shape[-1]-1)
    for b in range(grid_table.shape[0]):
        filename = docs[b][0][0]
        
        data_input_flat = grid_table[b,:,:,0].reshape([-1])
        labels = gt_classes[b,:,:].reshape([-1])
        logits = model_output_val[b,:,:,:].reshape([-1, data_loader.num_classes])
        
        # ignore inputs that are not word
        indexes = np.where(data_input_flat != 0)[0]
        data_selected = data_input_flat[indexes]
        labels_selected = labels[indexes]
        logits_array_selected = logits[indexes]
        
        # calculate accuracy
        for c in range(1, data_loader.num_classes):
            labels_indexes = np.where(labels_selected == c)[0]
            logits_indexes = np.where(logits_array_selected[:,c] > params.c_threshold)[0]
            if np.array_equal(labels_indexes, logits_indexes): 
                num_correct_strict += 1        
            try:  
                num_correct += np.shape(np.intersect1d(labels_indexes, logits_indexes))[0] / np.shape(labels_indexes)[0]
            except ZeroDivisionError:
                if np.shape(logits_indexes)[0] == 0:
                    num_correct += 1
                else:
                    num_correct += 0                    
        
            # show results without the <DontCare> class      
            res += '\n{}(GT/Inf):\t"'.format(data_loader.classes[c])
            
            # ground truth label
            gt = str(' '.join([data_loader.index_to_word[i] for i in data_selected[labels_indexes]]).encode('utf-8'))
            predict = str(' '.join([data_loader.index_to_word[i] for i in data_selected[logits_indexes]]).encode('utf-8'))
            res += gt + '" | "' + predict + '"'
        
            # write results to csv
            fieldnames = ['TaskID', 'GT', 'Predicted']
            
            csv_filename = 'data/results/' + params.save_prefix + '_' + data_loader.classes[c] + '.csv'            
            writer = csv.DictWriter(open(csv_filename, 'a'), fieldnames=fieldnames) 
            row = {'TaskID':filename, 'GT':gt, 'Predicted':predict}
            writer.writerow(row)
            
            csv_diff_filename = 'data/results/' + params.save_prefix + '_Diff_' + data_loader.classes[c] + '.csv'
            if gt != predict:
                writer = csv.DictWriter(open(csv_diff_filename, 'a'), fieldnames=fieldnames) 
                row = {'TaskID':filename, 'GT':gt, 'Predicted':predict}
                writer.writerow(row)
            
            # wrong inferences results
            if not np.array_equal(labels_indexes, logits_indexes): 
                res += '"\n \t FALSES =>>'
                logits_flat = logits_array_selected[:,c]
                fault_logits_indexes = np.setdiff1d(logits_indexes, labels_indexes)
                for i in range(len(data_selected)):
                    if i not in fault_logits_indexes: # only show fault_logits_indexes
                        continue
                    w = data_loader.index_to_word[data_selected[i]]
                    l = data_loader.classes[labels_selected[i]]
                    res += ' "%s"/%s, '%(w, l)
                    #res += ' "%s"/%.2f%s, '%(w, logits_flat[i], l)
                        
            #print(res)
    recall = num_correct / num_all
    accuracy_strict = num_correct_strict / num_all
    return recall, accuracy_strict, res

if __name__ == '__main__':
    # data
    #data_loader = DataLoader(params, True, True) # True to use 25% training data
    data_loader = DataLoader(params, update_dict=False, load_dictionary=True, data_split=0) # False to provide a path with only test data
    num_words = max(40000, data_loader.num_words)
    num_classes = data_loader.num_classes

    # model
    network = CUTIERes(num_words, num_classes, params, False)   
    
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
        recalls, accs_strict = [], []
        data = data_loader.fetch_test_data()        
        model_output = network.inference()    
        while data:
            print('{:d} samples left to be tested'.format(len(data['grid_table'])))
            grid_tables, gt_classes = [], []
            if len(data['grid_table']) > params.batch_size:
                docs = data['docs'][:params.batch_size]
                grid_tables = data['grid_table'][:params.batch_size]
                gt_classes = data['gt_classes'][:params.batch_size]
                del data['grid_table'][:params.batch_size]
                del data['gt_classes'][:params.batch_size]
            else:
                grid_tables = data['grid_table'][:]
                gt_classes = data['gt_classes'][:]
                data = None
            feed_dict = {
                network.data: grid_tables
            }
            fetches = [model_output]
            
            [model_output_val] = sess.run(fetches=fetches, feed_dict=feed_dict)                    
            recall, acc_strict, res = cal_save_results(np.array(docs), np.array(grid_tables), np.array(gt_classes), model_output_val)
            recalls += [recall]
            accs_strict += [acc_strict] 
            print(res) # show res for current batch    

        recall = sum(recalls) / len(recalls)
        acc_strict = sum(accs_strict) / len(accs_strict)
        print('VALIDATION ACC (Recall/Acc): %.3f / %.3f\n'%(recall, acc_strict))