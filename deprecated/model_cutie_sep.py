# written by Xiaohui Zhao
# 2018-12 
# xiaohui.zhao@accenture.com
import tensorflow as tf
from model_cutie import CUTIE    
    
class CUTIESep(CUTIE):
    def __init__(self, num_vocabs, num_classes, params, trainable=True):
        self.name = "CUTIE_seperatable_residual"
        
        self.data = tf.placeholder(tf.int32, shape=[None, None, None, 1], name='grid_table')
        self.gt_classes = tf.placeholder(tf.int32, shape=[None, None, None], name='gt_classes')
        self.layers = dict({'data': self.data, 'gt_classes': self.gt_classes})  
        self.num_vocabs = num_vocabs
        self.num_classes = num_classes     
        self.trainable = trainable
        
        self.embedding_size = params.embedding_size
        self.weight_decay = params.weight_decay if hasattr(params, 'weight_decay') else 0
        self.hard_negative_ratio = params.hard_negative_ratio if hasattr(params, 'hard_negative_ratio') else 0
        self.batch_size = params.batch_size if hasattr(params, 'batch_size') else 0
        
        self.layer_inputs = []        
        self.setup()
        
    
    def setup(self):        
        # input
        (self.feed('data')
             .embed(self.num_vocabs, self.embedding_size, name='embedding'))  
        
        # encoder
        (self.feed('embedding')
             .sepconv(3, 5, 4, 2, name='encoder1_1')
             .sepconv(3, 5, 4, 2, name='encoder1_2')
             .conv(3, 5, 128, 1, 1, name='bottleneck1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .sepconv(3, 5, 4, 2, name='encoder2_1')
             .sepconv(3, 5, 4, 2, name='encoder2_2')
             .conv(3, 5, 256, 1, 1, name='bottleneck2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .sepconv(3, 5, 4, 2, name='encoder3_1')
             .sepconv(3, 5, 4, 2, name='encoder3_2')
             .conv(3, 5, 512, 1, 1, name='bottleneck3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .sepconv(3, 5, 4, 2, name='encoder4_1')
             .sepconv(3, 5, 4, 2, name='encoder4_2'))
        
        # decoder
        (self.feed('encoder4_2')
             .up_conv(3, 5, 512, 1, 1, name='up1')
             .conv(3, 5, 256, 1,1, name='bottleneck4')
             .sepconv(3, 5, 4, 2, name='decoder1_1')
             .sepconv(3, 5, 4, 2, name='decoder1_2')
             .up_conv(3, 5, 256, 1, 1, name='up2')
             .conv(3, 5, 128, 1,1, name='bottleneck5')
             .sepconv(3, 5, 4, 2, name='decoder2_1')
             .sepconv(3, 5, 4, 2, name='decoder2_2')
             .up_conv(3, 5, 128, 1, 1, name='up3')
             .conv(3, 5, 64, 1,1, name='bottleneck6')
             .sepconv(3, 5, 4, 2, name='decoder3_1')
             .sepconv(3, 5, 4, 2, name='decoder3_2'))
        
        # classification
        (self.feed('decoder3_2')
             .conv(1, 1, self.num_classes, 1, 1, name='cls_logits')
             .softmax(name='softmax'))  