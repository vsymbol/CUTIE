# written by Xiaohui Zhao
# 2019-03
# xiaohui.zhao@accenture.com
import tensorflow as tf
from model_cutie import CUTIE    
    
class CUTIERes(CUTIE):
    def __init__(self, num_vocabs, num_classes, params, trainable=True):
        self.name = "CUTIE_attention" # 
        
        self.data = tf.placeholder(tf.int32, shape=[None, None, None, 1], name='grid_table')
        self.gt_classes = tf.placeholder(tf.int32, shape=[None, None, None], name='gt_classes')
        self.use_ghm = tf.equal(1, params.use_ghm) if hasattr(params, 'use_ghm') else tf.equal(1, 0) #params.use_ghm 
        self.activation = 'sigmoid' if (hasattr(params, 'use_ghm') and params.use_ghm) else 'relu'
        self.ghm_weights = tf.placeholder(tf.float32, shape=[None, None, None, num_classes], name='ghm_weights')        
        self.layers = dict({'data': self.data, 'gt_classes': self.gt_classes, 'ghm_weights':self.ghm_weights})

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
             .embed(self.num_vocabs, self.embedding_size, name='embedding')
             .conv(3, 5, 128, 1, 1, name='encoder0_1'))  
        
        # encoder
        (self.feed('encoder0_1')
             .conv(3, 5, 128, 1, 1, name='encoder1_1')
             .conv(3, 5, 128, 1, 1, name='encoder1_2')
             .conv(3, 5, 128, 1, 1, name='encoder1_3')
             .conv(3, 5, 128, 1, 1, name='encoder1_4'))
        
        (self.feed('encoder0_1', 'encoder1_4')
             .attention(1, name='attention2')
             .conv(3, 5, 128, 1, 1, name='encoder1_5')
             .conv(3, 5, 128, 1, 1, name='encoder1_6')
             .conv(3, 5, 128, 1, 1, name='encoder1_7')
             .conv(3, 5, 128, 1, 1, name='encoder1_8')) 
        
        (self.feed('encoder0_1', 'encoder1_8')
             .attention(1, name='attention5')
             .conv(3, 5, 128, 1, 1, name='encoder1_9')
             .conv(3, 5, 128, 1, 1, name='encoder1_10')) 
         
        # classification
        (self.feed('encoder1_10') 
             .conv(1, 1, self.num_classes, 1, 1, activation=self.activation, name='cls_logits') # sigmoid for ghm
             .softmax(name='softmax'))