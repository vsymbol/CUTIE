# written by Xiaohui Zhao
# 2018-12 
# xiaohui.zhao@accenture.com
import tensorflow as tf
import numpy as np
import argparse, os
import timeit
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model_cutie import CUTIE
from model_cutie_res_bert import CUTIERes
from model_cutie_unet8 import CUTIEUNet
from data_loader_json import DataLoader
from utils import *

parser = argparse.ArgumentParser(description='CUTIE parameters')
# data
parser.add_argument('--doc_path', type=str, default='data/meals') 
parser.add_argument('--save_prefix', type=str, default='meals', help='prefix for ckpt') # TBD: save log/models with prefix

# ckpt
parser.add_argument('--restore_ckpt', type=bool, default=True) 
parser.add_argument('--restore_bertembedding_only', type=bool, default=True) # effective when restore_ckpt is True
parser.add_argument('--embedding_file', type=str, default='../graph/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt') 
parser.add_argument('--ckpt_path', type=str, default='../graph/CUTIE/graph/')
parser.add_argument('--ckpt_file', type=str, default='CUTIE_residual_8x_40000x9_iter_10000.ckpt')  

# dict
parser.add_argument('--load_dict', type=bool, default=True, help='True to work based on an existing dict') 
parser.add_argument('--load_dict_from_path', type=str, default='dict/119547') # 40000 or 119547  
parser.add_argument('--update_dict', type=bool, default=True) 
parser.add_argument('--dict_path', type=str, default='dict/---') # not used if load_dict is True

# log
parser.add_argument('--log_path', type=str, default='../graph/CUTIE/log/') 
parser.add_argument('--log_disp_step', type=int, default=100) 
parser.add_argument('--log_save_step', type=int, default=100) 
parser.add_argument('--validation_step', type=int, default=200) 
parser.add_argument('--ckpt_save_step', type=int, default=1000)

# training
parser.add_argument('--data_augmentation', type=bool, default=True) # augment data row/col in each batch
parser.add_argument('--data_augmentation_extra', type=bool, default=True) # randomly expand rows/cols
parser.add_argument('--data_augmentation_extra_rows', type=int, default=4) 
parser.add_argument('--data_augmentation_extra_cols', type=int, default=8) 
parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--iterations', type=int, default=80000)  
parser.add_argument('--lr_decay_step', type=int, default=4000) 
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5) 

# loss optimization
parser.add_argument('--hard_negative_ratio', type=int, help='the ratio between negative and positive losses', default=3) 
parser.add_argument('--use_ghm', type=int, default=0) # 1 to use GHM, 0 to not use
parser.add_argument('--ghm_bins', type=int, default=30) # to be tuned
parser.add_argument('--ghm_momentum', type=int, default=0) # 0 / 0.75

# model
parser.add_argument('--embedding_size', type=int, default=120) # not used for bert embedding with default 768
parser.add_argument('--weight_decay', type=float, default=0.0005) 
parser.add_argument('--eps', type=float, default=1e-6) 

# inference
parser.add_argument('--c_threshold', type=float, default=0.5) 
params = parser.parse_args()

edges = [float(x)/params.ghm_bins for x in range(params.ghm_bins+1)]
edges[-1] += params.eps
acc_sum = [0.0 for _ in range(params.ghm_bins)]
def calc_ghm_weights(logits, labels): 
    """
    calculate gradient harmonizing mechanism weights
    """
    bins = params.ghm_bins
    momentum = params.ghm_momentum   
    shape = logits.shape     
    
    logits_flat = logits.reshape([-1])
    labels_flat = labels.reshape([-1])
    arr = [0 for _ in range(len(labels_flat)*num_classes)]
    for i,l in enumerate(labels_flat):
        arr[i*num_classes + l] = 1
    labels_flat = np.array(arr)
    
    grad = abs(logits_flat - labels_flat) # equation for logits from the sigmoid activation
    
    weights = np.ones(logits_flat.shape)
    N = shape[0] * shape[1] * shape[2] * shape[3]
    M = 0
    for i in range(bins):
        idxes = np.multiply(grad>=edges[i], grad<edges[i+1])
        num_in_bin = np.sum(idxes)
        if num_in_bin > 0: 
            acc_sum[i] = momentum * acc_sum[i] + (1-momentum) * num_in_bin
            weights[np.where(idxes)] = N / acc_sum[i]
            M += 1
    if M > 0:
        weights = weights / M
        
    return weights.reshape(shape)

if __name__ == '__main__':
    # data
    data_loader = DataLoader(params, update_dict=params.update_dict, load_dictionary=params.load_dict, data_split=0.75)
    num_words = max(40000, data_loader.num_words)
    num_classes = data_loader.num_classes
    #a = data_loader.next_batch()
    #b = data_loader.fetch_validation_data()
    
    # model
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
    
    config = tf.ConfigProto(allow_soft_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        iter_start = 0
        
        # restore parameters
        if params.restore_ckpt:
            if params.restore_bertembedding_only:
                if 'bert' not in network.name:
                    raise Exception('no bert embedding was designed in the built model, \
                        switch restore_bertembedding_only off or built a related model')
                try:
                    load_variable = {"bert/embeddings/word_embeddings": network.embedding_table}
                    ckpt_saver = tf.train.Saver(load_variable, max_to_keep=50)
                    ckpt_path = params.embedding_file
                    ckpt = tf.train.get_checkpoint_state(ckpt_path)
                    print('Restoring from {}...'.format(ckpt_path))
                    ckpt_saver.restore(sess, ckpt_path)
                    print('Restored from {}'.format(ckpt_path))
                except:
                    raise Exception('Check your path {:s}'.format(ckpt_path))
            else:
                try:
                    ckpt_path = os.path.join(params.ckpt_path, params.ckpt_file)
                    ckpt = tf.train.get_checkpoint_state(ckpt_path)
                    print('Restoring from {}...'.format(ckpt_path))
                    ckpt_saver.restore(sess, ckpt_path)
                    print('Restored from {}'.format(ckpt_path))
                    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
                    iter_start = int(stem.split('_')[-1]) - 1
                    sess.run(global_step.assign(iter_start))
                except:
                    raise Exception('Check your pretrained {:s}'.format(ckpt_path))
            
        # iterations
        for iter in range(iter_start, params.iterations):
            timer_start = timeit.default_timer()
            
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
                timer_stop = timeit.default_timer()
                print('\t >>time per step: %.2fs <<'%(timer_stop - timer_start))
                
                recall, acc_strict, res = cal_accuracy(data_loader, np.array(data['grid_table']), np.array(data['gt_classes']), model_output_val, params.c_threshold)
                training_recall += [recall]        
                training_acc_strict += [acc_strict]          
                print('\nIter: %d/%d, total loss: %.4f, model loss: %.4f, regularization loss: %.4f'%\
                      (iter, params.iterations, total_loss_val, model_loss_val, regularization_loss_val))
                print(res)
                print('TRAINING ACC (Recall/Acc): %.3f / %.3f | highest %.3f / %.3f'%(recall, acc_strict, max(training_recall), max(training_acc_strict)))
                
            # calculate validation accuracy and display results
            if (iter+1)%params.validation_step == 0:
                
                recalls, accs_strict = [], []
                for _ in range(params.batch_size):
                    data = data_loader.fetch_validation_data()
                    grid_tables = data['grid_table']
                    gt_classes = data['gt_classes']
                    
                    feed_dict = {
                        network.data: grid_tables
                    }
                    fetches = [model_output]                    
                    [model_output_val] = sess.run(fetches=fetches, feed_dict=feed_dict)                    
                    recall, acc_strict, res = cal_accuracy(data_loader, np.array(grid_tables), 
                                                           np.array(gt_classes), model_output_val, 
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
    
    from pprint import pprint
    pprint(params)
    summary_writer.close()
