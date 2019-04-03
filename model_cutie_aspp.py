# written by Xiaohui Zhao
# 2019-03
# xiaohui.zhao@accenture.com
import tensorflow as tf
from model_cutie import CUTIE    
    
class CUTIERes(CUTIE):
    def __init__(self, num_vocabs, num_classes, params, trainable=True):
        self.name = "CUTIE_atrousSPP" # 
        
        self.data_grid = tf.placeholder(tf.int32, shape=[None, None, None, 1], name='data_grid')
        self.gt_classes = tf.placeholder(tf.int32, shape=[None, None, None], name='gt_classes') 
        self.data_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data_image') # not used in CUTIEv1
        self.ps_1d_indices = tf.placeholder(tf.int32, shape=[None, None], name='ps_1d_indices') # not used in CUTIEv1
        
        self.use_ghm = tf.equal(1, params.use_ghm) if hasattr(params, 'use_ghm') else tf.equal(1, 0) #params.use_ghm 
        self.activation = 'sigmoid' if (hasattr(params, 'use_ghm') and params.use_ghm) else 'relu'
        self.dropout = params.data_augmentation_dropout if hasattr(params, 'data_augmentation_dropout') else 1
        self.ghm_weights = tf.placeholder(tf.float32, shape=[None, None, None, num_classes], name='ghm_weights')        
        self.layers = dict({'data_grid': self.data_grid, 'gt_classes': self.gt_classes, 'ghm_weights':self.ghm_weights})

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
        (self.feed('data_grid')
             .embed(self.num_vocabs, self.embedding_size, name='embedding', dropout=self.dropout))  
        
        # encoder
        (self.feed('embedding')
             .conv(3, 5, 256, 1, 1, name='encoder1_1')
             .conv(3, 5, 256, 1, 1, name='encoder1_2')
             .conv(3, 5, 256, 1, 1, name='encoder1_3')
             .conv(3, 5, 256, 1, 1, name='encoder1_4')
             .dilate_conv(3, 5, 256, 1, 1, 2, name='encoder1_5')
             .dilate_conv(3, 5, 256, 1, 1, 4, name='encoder1_6')
             .dilate_conv(3, 5, 256, 1, 1, 8, name='encoder1_7')
             .dilate_conv(3, 5, 256, 1, 1, 16, name='encoder1_8'))
        
        # Atrous Spatial Pyramid Pooling module
        #(self.feed('encoder1_8')
        #     .conv(1, 1, 256, 1, 1, name='aspp_0'))
        (self.feed('encoder1_8')
             .dilate_conv(3, 5, 256, 1, 1, 4, name='aspp_1'))
        (self.feed('encoder1_8')
             .dilate_conv(3, 5, 256, 1, 1, 8, name='aspp_2'))
        (self.feed('encoder1_8')
             .dilate_conv(3, 5, 256, 1, 1, 16, name='aspp_3'))
        (self.feed('encoder1_8')
             .global_pool(name='aspp_4'))
        (self.feed('aspp_1', 'aspp_2', 'aspp_3', 'aspp_4')
             .concat(3, name='aspp_concat')
             .conv(1, 1, 256, 1, 1, name='aspp_1x1'))
        
        # combine low level features
        (self.feed('encoder1_1', 'aspp_1x1')
             .concat(3, name='concat1')
             .conv(3, 5, 64, 1, 1, name='decoder1_1'))
        
        # classification
        (self.feed('decoder1_1') 
             .conv(1, 1, self.num_classes, 1, 1, activation=self.activation, name='cls_logits') # sigmoid for ghm
             .softmax(name='softmax'))