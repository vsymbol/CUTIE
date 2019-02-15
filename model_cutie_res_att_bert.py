# written by Xiaohui Zhao
# 2018-12 
# xiaohui.zhao@accenture.com
import tensorflow as tf
from model_cutie import CUTIE    
    
class CUTIERes(CUTIE):
    def __init__(self, num_vocabs, num_classes, params, trainable=True):
        self.name = "CUTIE_residual_attention_bert_8x" # 8x down sampling
        
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
        self.embedding_table = None      
        self.setup()
        
    
    def setup(self):        
        # input
        (self.feed('data')
             .bert_embed(self.num_vocabs, 768, name='embeddings', trainable=False))  
        
        # encoder
        (self.feed('embeddings')
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
             .up_conv(3, 5, 512, 1, 1, name='up1'))        
        (self.feed('up1', 'encoder3_2')
             .attention(1, name='attention1')
             .conv(3, 5, 256, 1, 1, name='decoder1_1')
             .conv(3, 5, 256, 1, 1, name='decoder1_2')
             .up_conv(3, 5, 256, 1, 1, name='up2'))       
        (self.feed('up2', 'encoder2_2')
             .attention(1, name='attention2')
             .conv(3, 5, 128, 1, 1, name='decoder2_1')
             .conv(3, 5, 128, 1, 1, name='decoder2_2')
             .up_conv(3, 5, 128, 1, 1, name='up3'))        
        (self.feed('up3', 'encoder1_2')
             .attention(1, name='attention3')
             .conv(3, 5, 64, 1, 1, name='decoder3_1')
             .conv(3, 5, 64, 1, 1, name='decoder3_2'))
        
        # classification
        (self.feed('decoder3_2') 
             .conv(1, 1, self.num_classes, 1, 1, activation=self.activation, name='cls_logits') # sigmoid for ghm
             .softmax(name='softmax'))