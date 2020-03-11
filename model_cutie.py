# written by Xiaohui Zhao
# 2018-12 
# xh.zhao@outlook.com
import tensorflow as tf
from model_framework import Model
    
    
class CUTIE(Model):
    def __init__(self, num_vocabs, num_classes, params, trainable=True):
        self.name = "CUTIE_benchmark"
        
        self.data = tf.placeholder(tf.int32, shape=[None, None, None, 1], name='grid_table')
        self.gt_classes = tf.placeholder(tf.int32, shape=[None, None, None], name='gt_classes')
        self.use_ghm = tf.equal(1, params.use_ghm) if hasattr(params, 'use_ghm') else tf.equal(1, 0) #params.use_ghm 
        self.activation = 'sigmoid' if (hasattr(params, 'use_ghm') and params.use_ghm) else 'relu'
        self.ghm_weights = tf.placeholder(tf.float32, shape=[None, None, None, num_classes], name='ghm_weights')        
        self.layers = dict({'data': self.data, 'gt_classes': self.gt_classes, 'ghm_weights': self.ghm_weights}) 
         
        self.num_vocabs = num_vocabs
        self.num_classes = num_classes     
        self.trainable = trainable
        
        self.embedding_size = params.embedding_size
        self.weight_decay = params.weight_decay if hasattr(params, 'weight_decay') else 0.0
        self.hard_negative_ratio = params.hard_negative_ratio if hasattr(params, 'hard_negative_ratio') else 0.0
        self.batch_size = params.batch_size if hasattr(params, 'batch_size') else 0
        
        self.layer_inputs = []        
        self.setup()
        
    
    def setup(self):        
        # input
        (self.feed('data')
             .embed(self.num_vocabs, self.embedding_size, name='embedding'))  
        
        # encoder
        (self.feed('embedding')
             .conv(3, 5, 64, 1, 1, name='encoder1_1')
             .conv(3, 5, 128, 1, 1, name='encoder1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 5, 128, 1, 1, name='encoder2_1')
             .conv(3, 5, 256, 1, 1, name='encoder2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 5, 256, 1, 1, name='encoder3_1')
             .conv(3, 5, 512, 1, 1, name='encoder3_2')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 5, 512, 1, 1, name='encoder4_1')
             .conv(3, 5, 512, 1, 1, name='encoder4_2'))
        
        # decoder
        (self.feed('encoder4_2')
             .up_conv(3, 5, 512, 1, 1, name='up1')
             .conv(3, 5, 256, 1, 1, name='decoder1_1')
             .conv(3, 5, 256, 1, 1, name='decoder1_2')
             .up_conv(3, 5, 256, 1, 1, name='up2')
             .conv(3, 5, 128, 1, 1, name='decoder2_1')
             .conv(3, 5, 128, 1, 1, name='decoder2_2')
             .up_conv(3, 5, 128, 1, 1, name='up3')
             .conv(3, 5, 64, 1, 1, name='decoder3_1')
             .conv(3, 5, 64, 1, 1, name='decoder3_2'))
        
        # classification
        (self.feed('decoder3_2')
             .conv(1, 1, self.num_classes, 1, 1, activation=self.activation, name='cls_logits')
             .softmax(name='softmax'))  
    
    def disp_results(self, data_input, data_label, model_output, threshold):
        data_input_flat = data_input.reshape([-1]) # [b * h * w]
        labels = [] # [b * h * w, classes]
        for item in data_label.reshape([-1]):
            labels.append([i==item for i in range(self.num_classes)])
        logits = model_output.reshape([-1, self.num_classes]) # [b * h * w, classes] 
        
        # ignore none word input
        labels_flat = []
        results_flat = []
        for idx, item in enumerate(data_input_flat):
            if item != 0: 
                labels_flat.extend(labels[idx])
                results_flat.extend(logits[idx] > threshold)
        
        num_p = sum(labels_flat)
        num_n = sum([1-label for label in labels_flat])   
        num_all = len(results_flat)     
        num_correct = sum([True for i in range(num_all) if labels_flat[i] == results_flat[i]])        
        
        labels_flat_p = [label!=0 for label in labels_flat]
        labels_flat_n = [label==0 for label in labels_flat]
        num_tp = sum([labels_flat_p[i] * results_flat[i] for i in range(num_all)])
        num_tn = sum([labels_flat_n[i] * (not results_flat[i]) for i in range(num_all)])
        num_fp = num_n - num_tp
        num_fn = num_p - num_tp
        
        # accuracy, precision, recall
        accuracy = num_correct / num_all
        precision = num_tp / (num_tp + num_fp)
        recall = num_tp / (num_tp + num_fn)
        
        return accuracy, precision, recall
        
        
    def inference(self):
        return self.get_output('softmax') #cls_logits
        
    
    def build_loss(self):
        labels = self.get_output('gt_classes')
        cls_logits = self.get_output('cls_logits')         
        cls_logits = tf.cond(self.use_ghm, lambda: cls_logits*self.get_output('ghm_weights'), 
                             lambda: cls_logits, name="GradientHarmonizingMechanism")      
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=cls_logits)
            
        with tf.variable_scope('HardNegativeMining'):
            labels = tf.reshape(labels, [-1])  
            cross_entropy = tf.reshape(cross_entropy, [-1])
            
            fg_idx = tf.where(tf.not_equal(labels, 0))
            fgs = tf.gather(cross_entropy, fg_idx)
            bg_idx = tf.where(tf.equal(labels, 0))
            bgs = tf.gather(cross_entropy, bg_idx)
             
            num = self.hard_negative_ratio * tf.shape(fgs)[0]
            num_bg = tf.cond(tf.shape(bgs)[0]<num, lambda:tf.shape(bgs)[0], lambda:num)
            sorted_bgs, _ = tf.nn.top_k(tf.transpose(bgs), num_bg, sorted=True)
            cross_entropy = fgs + sorted_bgs
        
        # total loss
        model_loss = tf.reduce_mean(cross_entropy)
        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='regularization')
        total_loss = model_loss + regularization_loss
        
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('regularization_loss', regularization_loss)
        tf.summary.scalar('total_loss', total_loss)
        
        logits = self.get_output('cls_logits')
        softmax_logits = self.get_output('softmax') #cls_logits
        return model_loss, regularization_loss, total_loss, logits, softmax_logits 
    
    def build_multi_loss(self):
        labels = self.get_output('gt_classes')
        cls_logits = self.get_output('cls_logits')
        
