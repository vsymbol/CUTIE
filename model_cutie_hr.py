# written by Xiaohui Zhao
# 2018-12 
# xiaohui.zhao@accenture.com
import tensorflow as tf
from model_cutie import CUTIE    
    
class CUTIERes(CUTIE):
    def __init__(self, num_vocabs, num_classes, params, trainable=True):
        self.name = "CUTIE_highresolution_8x" # 8x down sampling
        
        self.data_grid = tf.placeholder(tf.int32, shape=[None, None, None, 1], name='grid_table')
        self.gt_classes = tf.placeholder(tf.int32, shape=[None, None, None], name='gt_classes')
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
        
        # stage 1 block 1
        (self.feed('embedding')
             .conv(3, 5, 64, 1, 1, name='encoder1_1_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_1_2')
             .conv(3, 5, 128, 2, 2, name='down1_1_2'))
        
        
        ## introduce stage 2
        # stage 1 block 2
        (self.feed('encoder1_1_2')
             .conv(3, 5, 64, 1, 1, name='encoder1_2_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_2_2')
             .conv(3, 5, 128, 2, 2, name='down1_2_2'))        
        # stage 2 block 2
        (self.feed('down1_1_2')
             .conv(3, 5, 128, 1, 1, name='encoder2_2_1')
             .conv(3, 5, 128, 1, 1, name='encoder2_2_2')
             .up_conv(3, 5, 64, 1, 1, factor=2, name='up2_2_1'))
        
        
        # stage 1 block 3
        (self.feed('encoder1_1_2', 'encoder1_2_2', 'up2_2_1')
             .add(name='add1_3')
             .conv(3, 5, 64, 1, 1, name='encoder1_3_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_3_2')
             .conv(3, 5, 128, 2, 2, name='down1_3_2'))
        (self.feed('encoder1_3_2')
             .conv(3, 5, 256, 4, 4, name='down1_3_3'))        
        # stage 2 block 3
        (self.feed('down1_1_2', 'encoder2_2_2', 'down1_2_2')
             .add(name='add2_3')
             .conv(3, 5, 128, 1, 1, name='encoder2_3_1')
             .conv(3, 5, 128, 1, 1, name='encoder2_3_2')
             .up_conv(3, 5, 64, 1, 1, factor=2, name='up2_3_1'))
        (self.feed('encoder2_3_2')
             .conv(3, 5, 256, 2, 2, name='down2_3_3'))
        
        
        ## introduce stage 3
        # stage 1 block 4
        (self.feed('add1_3', 'encoder1_3_2', 'up2_3_1')
             .add(name='add1_4')
             .conv(3, 5, 64, 1, 1, name='encoder1_4_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_4_2')
             .conv(3, 5, 128, 2, 2, name='down1_4_2'))
        (self.feed('encoder1_4_2')
             .conv(3, 5, 256, 4, 4, name='down1_4_3'))        
        # stage 2 block 4
        (self.feed('add2_3', 'encoder2_3_2', 'down1_3_2')
             .add(name='add2_4')
             .conv(3, 5, 128, 1, 1, name='encoder2_4_1')
             .conv(3, 5, 128, 1, 1, name='encoder2_4_2')
             .up_conv(3, 5, 64, 1, 1, name='up2_4_1'))  
        (self.feed('encoder2_4_2')
             .conv(3, 5, 256, 2, 2, name='down2_4_3'))        
        # stage 3 block 4
        (self.feed('down1_3_3', 'down2_3_3')
             .add(name='add3_4')
             .conv(3, 5, 256, 1, 1, name='encoder3_4_1')
             .conv(3, 5, 256, 1, 1, name='encoder3_4_2')
             .up_conv(3, 5, 64, 1, 1, factor=4, name='up3_4_1'))
        (self.feed('encoder3_4_2')
             .up_conv(3, 5, 128, 1, 1, factor=2, name='up3_4_2'))
        
        
        # stage 1 block 5
        (self.feed('add1_4', 'encoder1_4_2', 'up2_4_1', 'up3_4_1')
             .add(name='add1_5')
             .conv(3, 5, 64, 1, 1, name='encoder1_5_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_5_2')
             .conv(3, 5, 128, 2, 2, name='down1_5_2'))
        (self.feed('encoder1_5_2')
             .conv(3, 5, 256, 4, 4, name='down1_5_3'))  
        (self.feed('encoder1_5_2')
             .conv(3, 5, 512, 8, 8, name='down1_5_4'))         
        # stage 2 block 5
        (self.feed('add2_4', 'encoder2_4_2', 'down1_4_2', 'up3_4_2')
             .add(name='add2_5')
             .conv(3, 5, 128, 1, 1, name='encoder2_5_1')
             .conv(3, 5, 128, 1, 1, name='encoder2_5_2')
             .up_conv(3, 5, 64, 1, 1, factor=2, name='up2_5_1'))
        (self.feed('encoder2_5_2')
             .conv(3, 5, 256, 2, 2, name='down2_5_3'))
        (self.feed('encoder2_5_2')
             .conv(3, 5, 512, 4, 4, name='down2_5_4'))   
        # stage 3 block 5
        (self.feed('add3_4', 'encoder3_4_2', 'down1_4_3', 'down2_4_3')
             .add(name='add3_5')
             .conv(3, 5, 256, 1, 1, name='encoder3_5_1')
             .conv(3, 5, 256, 1, 1, name='encoder3_5_2')
             .up_conv(3, 5, 64, 1, 1, factor=4, name='up3_5_1'))
        (self.feed('encoder3_5_2')
             .up_conv(3, 5, 128, 1, 1, factor=2, name='up3_5_2'))
        (self.feed('encoder3_5_2')
             .conv(3, 5, 512, 2, 2, name='down3_5_4'))    
        
        
        ## introduce stage 4
        # stage 1 block 6
        (self.feed('add1_5', 'encoder1_5_2', 'up2_5_1', 'up3_5_1')
             .add(name='add1_6')
             .conv(3, 5, 64, 1, 1, name='encoder1_6_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_6_2')
             .conv(3, 5, 128, 2, 2, name='down1_6_2'))
        (self.feed('encoder1_6_2')
             .conv(3, 5, 256, 4, 4, name='down1_6_3'))
        (self.feed('encoder1_6_2')
             .conv(3, 5, 512, 8, 8, name='down1_6_4'))        
        # stage 2 block 6
        (self.feed('add2_5', 'encoder2_5_2', 'down1_5_2', 'up3_5_2')
             .add(name='add2_6')
             .conv(3, 5, 128, 1, 1, name='encoder2_6_1')
             .conv(3, 5, 128, 1, 1, name='encoder2_6_2')
             .up_conv(3, 5, 64, 1, 1, name='up2_6_1'))  
        (self.feed('encoder2_6_2')
             .conv(3, 5, 256, 2, 2, name='down2_6_3'))  
        (self.feed('encoder2_6_2')
             .conv(3, 5, 512, 4, 4, name='down2_6_4'))          
        # stage 3 block 6
        (self.feed('add3_5', 'encoder3_5_2', 'down1_5_3', 'down2_5_3')
             .add(name='add3_6')
             .conv(3, 5, 256, 1, 1, name='encoder3_6_1')
             .conv(3, 5, 256, 1, 1, name='encoder3_6_2')
             .up_conv(3, 5, 64, 1, 1, factor=4, name='up3_6_1'))
        (self.feed('encoder3_6_2')
             .up_conv(3, 5, 128, 1, 1, factor=2, name='up3_6_2'))
        (self.feed('encoder3_6_2')
             .conv(3, 5, 512, 2, 2, name='down3_6_4'))
        # stage 4 block 6
        (self.feed('down1_5_4', 'down2_5_4', 'down3_5_4')
             .add(name='add4_6')
             .conv(3, 5, 512, 1, 1, name='encoder4_6_1')
             .conv(3, 5, 512, 1, 1, name='encoder4_6_2')
             .up_conv(3, 5, 64, 1, 1, factor=8, name='up4_6_1'))
        (self.feed('encoder4_6_2')
             .up_conv(3, 5, 128, 1, 1, factor=4, name='up4_6_2'))
        (self.feed('encoder4_6_2')
             .up_conv(3, 5, 256, 1, 1, factor=2, name='up4_6_3'))
        
        
        # stage 1 block 7
        (self.feed('add1_6', 'encoder1_6_2', 'up2_6_1', 'up3_6_1', 'up4_6_1')
             .add(name='add1_7')
             .conv(3, 5, 64, 1, 1, name='encoder1_7_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_7_2')
             .conv(3, 5, 128, 2, 2, name='down1_7_2'))
        (self.feed('encoder1_7_2')
             .conv(3, 5, 256, 4, 4, name='down1_7_3'))
        (self.feed('encoder1_7_2')
             .conv(3, 5, 512, 8, 8, name='down1_7_4'))        
        # stage 2 block 7
        (self.feed('add2_6', 'encoder2_6_2', 'down1_6_2', 'up3_6_2', 'up4_6_2')
             .add(name='add2_7')
             .conv(3, 5, 128, 1, 1, name='encoder2_7_1')
             .conv(3, 5, 128, 1, 1, name='encoder2_7_2')
             .up_conv(3, 5, 64, 1, 1, factor=2, name='up2_7_1'))  
        (self.feed('encoder2_7_2')
             .conv(3, 5, 256, 2, 2, name='down2_7_3'))  
        (self.feed('encoder2_7_2')
             .conv(3, 5, 512, 4, 4, name='down2_7_4'))          
        # stage 3 block 7
        (self.feed('add3_6', 'encoder3_6_2', 'down1_6_3', 'down2_6_3', 'up4_6_3')
             .add(name='add3_7')
             .conv(3, 5, 256, 1, 1, name='encoder3_7_1')
             .conv(3, 5, 256, 1, 1, name='encoder3_7_2')
             .up_conv(3, 5, 64, 1, 1, factor=4, name='up3_7_1'))
        (self.feed('encoder3_7_2')
             .up_conv(3, 5, 128, 1, 1, factor=2, name='up3_7_2'))
        (self.feed('encoder3_7_2')
             .conv(3, 5, 512, 2, 2, name='down3_7_4'))        
        # stage 4 block 7
        (self.feed('add4_6', 'encoder4_6_2', 'down1_6_4', 'down2_6_4', 'down3_6_4')
             .add(name='add4_7')
             .conv(3, 5, 512, 1, 1, name='encoder4_7_1')
             .conv(3, 5, 512, 1, 1, name='encoder4_7_2')
             .up_conv(3, 5, 64, 1, 1, factor=8, name='up4_7_1'))
        (self.feed('encoder4_7_2')
             .up_conv(3, 5, 128, 1, 1, factor=4, name='up4_7_2'))
        (self.feed('encoder4_7_2')
             .up_conv(3, 5, 256, 1, 1, factor=2, name='up4_7_3'))
        
        
        # stage 1 block 8
        (self.feed('add1_7', 'encoder1_7_2', 'up2_7_1', 'up3_7_1', 'up4_7_1')
             .add(name='add1_8')
             .conv(3, 5, 64, 1, 1, name='encoder1_8_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_8_2'))        
        # stage 2 block 8
        (self.feed('add2_7', 'encoder2_7_2', 'down1_7_2', 'up3_7_2', 'up4_7_2')
             .add(name='add2_8')
             .conv(3, 5, 128, 1, 1, name='encoder2_8_1')
             .conv(3, 5, 128, 1, 1, name='encoder2_8_2')
             .up_conv(3, 5, 64, 1, 1, factor=2, name='up2_8_1'))          
        # stage 3 block 8
        (self.feed('add3_7', 'encoder3_7_2', 'down1_7_3', 'down2_7_3', 'up4_7_3')
             .add(name='add3_8')
             .conv(3, 5, 256, 1, 1, name='encoder3_8_1')
             .conv(3, 5, 256, 1, 1, name='encoder3_8_2')
             .up_conv(3, 5, 64, 1, 1, factor=4, name='up3_8_1'))        
        # stage 4 block 8
        (self.feed('add4_7', 'encoder4_7_2', 'down1_7_4', 'down2_7_4', 'down3_7_4')
             .add(name='add4_8')
             .conv(3, 5, 512, 1, 1, name='encoder4_8_1')
             .conv(3, 5, 512, 1, 1, name='encoder4_8_2')
             .up_conv(3, 5, 64, 1, 1, factor=8, name='up4_8_1'))
        
        
        # stage 1 block 9
        (self.feed('add1_8', 'encoder1_8_2', 'up2_8_1', 'up3_8_1', 'up4_8_1')
             .add(name='add1_9')
             .conv(3, 5, 64, 1, 1, name='encoder1_9_1')
             .conv(3, 5, 64, 1, 1, name='encoder1_9_2'))        
        
        
        # classification
        (self.feed('encoder1_9_2') 
             .conv(1, 1, self.num_classes, 1, 1, activation=self.activation, name='cls_logits') # sigmoid for ghm
             .softmax(name='softmax'))