# written by Xiaohui Zhao
# 2018-12 
# xiaohui.zhao@accenture.com
import tensorflow as tf
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model_cutie import CUTIE
from model_cutie_res import CUTIERes
from model_cutie_sep import CUTIESep
from data_loader import DataLoader

parser = argparse.ArgumentParser(description='CUTIE parameters')
parser.add_argument('--training_doc_path', type=str, default='data/training_data') 
parser.add_argument('--training_label_path', type=str, default='data/label') 
parser.add_argument('--validation_doc_path', type=str, default='data/validation_data') 
parser.add_argument('--validation_label_path', type=str, default='data/label') 

parser.add_argument('--restore_ckpt', type=bool, default=False) 
parser.add_argument('--ckpt_path', type=str, default='../graph/CUTIE/graph/')
parser.add_argument('--ckpt_file', type=str, default='CUTIE_residual_iter_9000.ckpt')  
parser.add_argument('--ckpt_save_prefix', type=str, default='CUTIE') # TBD: save log/models with prefix

parser.add_argument('--log_path', type=str, default='../graph/CUTIE/log/') 
parser.add_argument('--log_disp_step', type=int, default=50) 
parser.add_argument('--log_save_step', type=int, default=10) 
parser.add_argument('--validation_step', type=int, default=50) 
parser.add_argument('--ckpt_save_step', type=int, default=1000)

parser.add_argument('--hard_negative_ratio', type=int, help='the ratio between negative and positive losses', default=3) 
parser.add_argument('--embedding_size', type=int, default=120) 

parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--iterations', type=int, default=10000)  
parser.add_argument('--lr_decay_step', type=int, default=2000) 
parser.add_argument('--lr_decay_factor', type=float, default=0.5) 
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005) 
parser.add_argument('--eps', type=float, default=1e-14) 

parser.add_argument('--c_threshold', type=float, default=0.5) 
params = parser.parse_args()

def cal_accuracy(gird_table, gt_classes, model_output_val):
    #num_tp = 0
    #num_fn = 0
    num_correct = 0
    num_correct_strict = 0
    num_all = gird_table.shape[0] * (model_output_val.shape[-1]-1)
    for b in range(gird_table.shape[0]):
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
                num_correct += 0
                
            #num_tp += np.sum(logits_array_selected[labels_indexes])           
            
            # show results without the <DontCare> class                    
            if b==0:
                res = '{}(GT/Inf):\t"'.format(data_loader.classes[c])
                
                # ground truth label
                res += ' '.join(data_loader.index_to_word[i] for i in data_selected[labels_indexes])
                res += '" | "'
                res += ' '.join(data_loader.index_to_word[i] for i in data_selected[logits_indexes])
                
                # wrong inferences results
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
                        
                print(res)
    accuracy = num_correct / num_all
    accuracy_strict = num_correct_strict / num_all
    return accuracy, accuracy_strict

if __name__ == '__main__':
    # data
    data_loader = DataLoader(params)
    #data_loader.next_batch()
    #data_loader.fetch_validation_data()
    
    # model
    #network = CUTIE(data_loader.num_words, data_loader.num_classes, params)
    network = CUTIERes(data_loader.num_words, data_loader.num_classes, params)
    #network = CUTIESep(data_loader.num_words, data_loader.num_classes, params)
    model_loss, regularization_loss, total_loss, model_output = network.build_loss()  
    
    # operators
    global_step = tf.Variable(0, trainable=False)
    lr = tf.Variable(params.learning_rate, trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)
    tvars = tf.trainable_variables()
    grads = tf.gradients(total_loss, tvars)
    clipped_grads, norm = tf.clip_by_global_norm(grads, 10.0)
    train_op = optimizer.apply_gradients(list(zip(clipped_grads, tvars)), global_step=global_step) 
    
    tf.contrib.training.add_gradients_summaries(zip(clipped_grads, tvars))
    summary_op = tf.summary.merge_all()    
    
    # calculate number of parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(network.name, ': ', total_parameters/1000/1000, 'M parameters')

    # training
    max_training_acc = 0
    max_validation_acc = 0
    max_training_acc_strict = 0
    max_validation_acc_strict = 0
    ckpt_saver = tf.train.Saver(max_to_keep=50)
    summary_path = os.path.join(params.log_path, network.name)
    summary_writer = tf.summary.FileWriter(summary_path, tf.get_default_graph(), flush_secs=10)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        iter_start = 0
        if params.restore_ckpt:
            try:
                ckpt_path = os.path.join(params.ckpt_path, network_name, params.ckpt_file)
                ckpt = tf.train.get_checkpoint_state(ckpt_path)
                print('Restoring from {}...'.format(ckpt_path))
                ckpt_saver.restore(sess, ckpt_path)
                stem = os.path.splitext(os.path.basename(ckpt_path))[0]
                iter_start = int(stem.split('_'))[-1] - 1
                sess.run(global_step.assign(iter_start))
            except:
                raise('Check your pretrained {:s}'.format(ckpt_path))
            
        for iter in range(iter_start, params.iterations):
            # learning rate decay
            if iter!=0 and iter%params.lr_decay_step==0:
                sess.run(tf.assign(lr, lr.eval()*params.lr_decay_factor))
            
            # one step 
            data = data_loader.next_batch()
            feed_dict = {
                network.data: data['grid_table'],
                network.gt_classes: data['gt_classes'],
            }
            fetches = [model_loss, regularization_loss, total_loss, summary_op, train_op, model_output]
            (model_loss_val, regularization_loss_val, total_loss_val, summary_str, _, model_output_val) =\
                sess.run(fetches=fetches, feed_dict=feed_dict)
                                
            # calculate training accuracy and display results
            if iter%params.log_disp_step == 0: 
                acc, acc_strict = cal_accuracy(np.array(data['grid_table']), np.array(data['gt_classes']), model_output_val)  
                if acc > max_training_acc:
                    max_training_acc = acc          
                if acc_strict > max_training_acc_strict:
                    max_training_acc_strict = acc_strict          
                print('TRAINING ACC: %.2f / %.2f | highest %.2f / %.2f \n'%(acc, acc_strict, max_training_acc, max_training_acc_strict))
                
                print('\nIter: %d/%d, total loss: %.4f, model loss: %.4f, regularization loss: %.4f \n'%\
                      (iter+1, params.iterations, total_loss_val, model_loss_val, regularization_loss_val))
                
#                 acc, precision, recall = network.disp_results(np.array(data['grid_table']), 
#                                                               np.array(data['gt_classes']),
#                                                               model_output_val, params.c_threshold)
#                 print('acc: %.2f, precision: %.2f, recall: %.2f\n'%(acc, precision, recall))            
                
            # calculate validation accuracy and display results
            if iter%params.validation_step == 0:
                data = data_loader.fetch_validation_data()
                feed_dict = {
                    network.data: data['grid_table']
                }
                fetches = [model_output]
                [model_output_val] = sess.run(fetches=fetches, feed_dict=feed_dict)
                    
                acc, acc_strict = cal_accuracy(np.array(data['grid_table']), np.array(data['gt_classes']), model_output_val)  
                if acc > max_validation_acc:
                    max_validation_acc = acc          
                if acc_strict > max_validation_acc_strict:
                    max_validation_acc_strict = acc_strict     
                print('VALIDATION ACC: %.2f / %.2f | highest %.2f / %.2f \n'%(acc, acc_strict, max_validation_acc, max_validation_acc_strict))
                
            # save logs
            if iter%params.log_save_step == 0:
                summary_writer.add_summary(summary_str, iter)                
                
            # save checkpoints
            if (iter+1)%params.ckpt_save_step==0:
                ckpt_path = os.path.join(params.ckpt_path, network.name)
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                filename = os.path.join(ckpt_path, network.name + '_iter_{:d}'.format(iter+1) + '.ckpt')
                ckpt_saver.save(sess, filename)
                print('Checkpoint saved to: {:s}'.format(filename))
    
    summary_writer.close()