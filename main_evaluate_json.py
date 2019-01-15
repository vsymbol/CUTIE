# written by Xiaohui Zhao
# 2018-01 
# xiaohui.zhao@accenture.com
import tensorflow as tf
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model_cutie import CUTIE
from model_cutie_res import CUTIERes
from model_cutie_sep import CUTIESep
from data_loader_json import DataLoader

parser = argparse.ArgumentParser(description='CUTIE parameters')
parser.add_argument('--doc_path', type=str, default='data/meals') 
parser.add_argument('--save_prefix', type=str, default='meals') 

parser.add_argument('--embedding_size', type=int, default=120) 
parser.add_argument('--batch_size', type=int, default=64) 

parser.add_argument('--restore_ckpt', type=bool, default=True) 
parser.add_argument('--ckpt_path', type=str, default='../graph/CUTIE/graph/')
parser.add_argument('--ckpt_file', type=str, default='CUTIE_residual_10566x9_iter_1.ckpt')  
parser.add_argument('--c_threshold', type=float, default=0.5) 
params = parser.parse_args()

def cal_accuracy(docs, gird_table, gt_classes, model_output_val):
    
    for doc in docs:
        for item in doc:
            file_name = item[0]
            text = item[1]
            word_id = item[2]
            x_left, y_top, x_right, y_bottom = item[3][:]
            image_w, image_h = item[4][:]

    res = ''
    num_correct = 0
    num_correct_strict = 0
    num_all = gird_table.shape[0] * (model_output_val.shape[-1]-1)
    for b in range(gird_table.shape[0]):
        doc = docs[b]
        data_input_flat = gird_table[b,:,:,0].reshape([-1])
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
            if b==0:
                res += '\n{}(GT/Inf):\t"'.format(data_loader.classes[c])
                
                # ground truth label
                res += ' '.join(data_loader.index_to_word[i] for i in data_selected[labels_indexes])
                res += '" | "'
                res += ' '.join(data_loader.index_to_word[i] for i in data_selected[logits_indexes])
                res += '"'
                
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

def save_result(data, model_output):
    pass

if __name__ == '__main__':
    # data
    data_loader = DataLoader(params, True) # use 25% training data
    
    # model
    network = CUTIERes(10566, 9, params, False)
    model_output = network.inference()          

    # training
    max_validation_recall = 0
    max_validation_acc_strict = 0
    ckpt_saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        try:
            ckpt_path = os.path.join(params.ckpt_path, params.save_prefix, params.ckpt_file)
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            print('Restoring from {}...'.format(ckpt_path))
            ckpt_saver.restore(sess, ckpt_path)
            print('{} restored'.format(ckpt_path))
        except:
            raise('Check your pretrained {:s}'.format(ckpt_path))
            
        # calculate validation accuracy and display results
        data = data_loader.fetch_test_data()
        
        recalls, accs_strict = [], []
        while data:
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
            recall, acc_strict, res = cal_accuracy(np.array(docs), np.array(grid_tables), np.array(gt_classes), model_output_val)
            recalls += [recall]
            accs_strict += [acc_strict] 
            print(res) # show res for current batch    

        recall = sum(recalls) / len(recalls)
        acc_strict = sum(accs_strict) / len(accs_strict)
        if recall > max_validation_recall:
            max_validation_recall = recall          
        if acc_strict > max_validation_acc_strict:
            max_validation_acc_strict = acc_strict  
        print('VALIDATION ACC (Recall/Acc): %.2f / %.2f | highest %.2f / %.2f \n'%(recall, acc_strict, max_validation_recall, max_validation_acc_strict))                