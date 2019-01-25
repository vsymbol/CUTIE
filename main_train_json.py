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
from model_cutie_unet8 import CUTIEUNet
from data_loader_json import DataLoader

parser = argparse.ArgumentParser(description='CUTIE parameters')
# data
parser.add_argument('--doc_path', type=str, default='data/taxi') 
parser.add_argument('--save_prefix', type=str, default='taxi', help='prefix for ckpt') # TBD: save log/models with prefix

# dict
parser.add_argument('--dict_path', type=str, default='dict/---') # not used if load_dict is True
parser.add_argument('--load_dict', type=bool, default=True, help='True to work based on an existing dict') 
parser.add_argument('--load_dict_from_path', type=str, default='dict/40000') 
parser.add_argument('--large_dict', type=bool, default=True, help='True to use large dict for future ext') 

# ckpt
parser.add_argument('--restore_ckpt', type=bool, default=False) 
parser.add_argument('--restore_embedding_only', type=bool, default=True) 
parser.add_argument('--ckpt_path', type=str, default='../graph/CUTIE/graph/meals/')
parser.add_argument('--ckpt_file', type=str, default='CUTIE_residual_16x_40000x9_iter_10000.ckpt')  

# log
parser.add_argument('--log_path', type=str, default='../graph/CUTIE/log/') 
parser.add_argument('--log_disp_step', type=int, default=50) 
parser.add_argument('--log_save_step', type=int, default=100) 
parser.add_argument('--validation_step', type=int, default=200) 
parser.add_argument('--ckpt_save_step', type=int, default=1000)

# loss optimization
parser.add_argument('--hard_negative_ratio', type=int, help='the ratio between negative and positive losses', default=3) 
parser.add_argument('--use_ghm', type=int, default=1) # 1 or 0
parser.add_argument('--ghm_bins', type=int, default=30) 
parser.add_argument('--ghm_momentum', type=int, default=0) 

# training
parser.add_argument('--embedding_size', type=int, default=120) 
parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--iterations', type=int, default=10000)  
parser.add_argument('--lr_decay_step', type=int, default=1500) 
parser.add_argument('--lr_decay_factor', type=float, default=0.5) 
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005) 
parser.add_argument('--eps', type=float, default=1e-6) 

# inference
parser.add_argument('--c_threshold', type=float, default=0.5) 
params = parser.parse_args()

if __name__ == '__main__':
    # data
    data_loader = DataLoader(params, for_train=True, load_dictionary=params.load_dict, data_split=0.75)
    num_words = 40000 if params.large_dict else data_loader.num_words
    num_classes = data_loader.num_classes
    #a = data_loader.next_batch()
    #b = data_loader.fetch_validation_data()
    
    # model
    #network = CUTIE(num_words, num_classes, params
    network = CUTIERes(num_words, num_classes, params)
    #network = CUTIEUNet(num_words, num_classes, params)
    model_loss, regularization_loss, total_loss, model_logits, model_output = network.build_loss()  
    
    # operators
    global_step = tf.Variable(0, trainable=False)
    lr = tf.Variable(params.learning_rate, trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)
    tvars = tf.trainable_variables()
    grads = tf.gradients(total_loss, tvars)
    clipped_grads, norm = tf.clip_by_global_norm(grads, 10.0)
    train_op = optimizer.apply_gradients(list(zip(clipped_grads, tvars)), global_step=global_step) 
    with tf.control_dependencies([train_op]):
        train_dummy = tf.constant(0)
  
    tf.contrib.training.add_gradients_summaries(zip(clipped_grads, tvars))
    summary_op = tf.summary.merge_all()    
    
    # calculate the number of parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(network.name, ': ', total_parameters/1000/1000, 'M parameters \n')

    # training
    training_recall = []
    validation_recall = []
    training_acc_strict = []
    validation_acc_strict = []
    
    ckpt_saver = tf.train.Saver(max_to_keep=50)
    summary_path = os.path.join(params.log_path, params.save_prefix, network.name)
    summary_writer = tf.summary.FileWriter(summary_path, tf.get_default_graph(), flush_secs=10)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        iter_start = 0
        if params.restore_ckpt:
            try:
                ckpt_path = os.path.join(params.ckpt_path, params.ckpt_file)
                ckpt = tf.train.get_checkpoint_state(ckpt_path)
                print('Restoring from {}...'.format(ckpt_path))
                ckpt_saver.restore(sess, ckpt_path)
                print('Restored from {}'.format(ckpt_path))
                #stem = os.path.splitext(os.path.basename(ckpt_path))[0]
                #iter_start = int(stem.split('_')[-1]) - 1
                #sess.run(global_step.assign(iter_start))
            except:
                raise('Check your pretrained {:s}'.format(ckpt_path))
            
        for iter in range(iter_start, params.iterations):
            # learning rate decay
            if iter!=0 and iter%params.lr_decay_step==0:
                sess.run(tf.assign(lr, lr.eval()*params.lr_decay_factor))
            
            data = data_loader.next_batch()
            feeds = [network.data, network.gt_classes, network.ghm_weights]
            fetches = [model_loss, regularization_loss, total_loss, summary_op, train_dummy, model_logits, model_output]
            h = sess.partial_run_setup(fetches, feeds)
            
            # one step inference 
            feed_dict = {
                network.data: data['grid_table'],
                network.gt_classes: data['gt_classes'],
            }
            fetches = [model_logits, model_output]
            (model_logit_val, model_output_val) = sess.partial_run(h, fetches, feed_dict)
            
            # one step training
            ghm_weights = np.ones(np.shape(model_logit_val))
            if params.use_ghm:
                ghm_weights = calc_ghm_weights(np.array(model_logit_val), np.array(data['gt_classes']))
            feed_dict = {
                network.ghm_weights: ghm_weights,
            }
            fetches = [model_loss, regularization_loss, total_loss, summary_op, train_dummy]
            (model_loss_val, regularization_loss_val, total_loss_val, summary_str, _) =\
                sess.partial_run(h, fetches=fetches, feed_dict=feed_dict)
                                
            # calculate training accuracy and display results
            if iter%params.log_disp_step == 0: 
                recall, acc_strict, res = cal_accuracy(np.array(data['grid_table']), 
                                                       np.array(data['gt_classes']), 
                                                       model_output_val, params.c_threshold)
                training_recall += [recall]        
                training_acc_strict += [acc_strict]          
                print('\nIter: %d/%d, total loss: %.4f, model loss: %.4f, regularization loss: %.4f'%\
                      (iter, params.iterations, total_loss_val, model_loss_val, regularization_loss_val))
                print(res)
                print('TRAINING ACC (Recall/Acc): %.3f / %.3f | highest %.3f / %.3f'%(recall, acc_strict, max(training_recall), max(training_acc_strict)))
                
            # calculate validation accuracy and display results
            if (iter+1)%params.validation_step == 0:
                data = data_loader.fetch_validation_data()
                
                recalls, accs_strict = [], []
                while data:
                    grid_tables, gt_classes = [], []
                    if len(data['grid_table']) > params.batch_size:
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
                    recall, acc_strict, res = cal_accuracy(np.array(grid_tables), 
                                                           np.array(gt_classes),
                                                           model_output_val, 
                                                           params.c_threshold)
                    recalls += [recall]
                    accs_strict += [acc_strict] 

                recall = sum(recalls) / len(recalls)
                acc_strict = sum(accs_strict) / len(accs_strict)
                validation_recall += [recall]
                validation_acc_strict += [acc_strict]  
                print(res) # show res from the last execution of the while loop    
                print('VALIDATION ACC (Recall/Acc): %.3f / %.3f | highest %.3f / %.3f \n'
                      %(recall, acc_strict, max(validation_recall), max(validation_acc_strict)))

                print('TRAINING ACC CURVE: ' 
                      + ' >'.join(['{:d}:{:.3f}'.
                                  format(i*params.log_disp_step,w) for i,w in enumerate(training_acc_strict)]))
                print('VALIDATION ACC CURVE: ' 
                      + ' >'.join(['{:d}:{:.3f}'.
                                  format(i*params.validation_step,w) for i,w in enumerate(validation_acc_strict)]))
                print('TRAINING RECALL CURVE: ' 
                      + ' >'.join(['{:d}:{:.2f}'.
                                  format(i*params.log_disp_step,w) for i,w in enumerate(training_recall)]))
                print('VALIDATION RECALL CURVE: ' 
                      + ' >'.join(['{:d}:{:.2f}'.
                                  format(i*params.validation_step,w) for i,w in enumerate(validation_recall)]))
                
            # save logs
            if (iter+1)%params.log_save_step == 0:
                summary_writer.add_summary(summary_str, iter+1)                
                
            # save checkpoints
            if (iter+1)%params.ckpt_save_step == 0:
                ckpt_path = os.path.join(params.ckpt_path, params.save_prefix)
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                filename = os.path.join(ckpt_path, network.name + '_{:d}x{:d}_iter_{:d}'.format(num_words, num_classes, iter+1) + '.ckpt')
                ckpt_saver.save(sess, filename)
                print('Checkpoint saved to: {:s}'.format(filename))
    
    summary_writer.close()