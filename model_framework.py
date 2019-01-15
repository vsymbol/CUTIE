# written by Xiaohui Zhao
# 2018-12 
# xiaohui.zhao@accenture.com
import tensorflow as tf


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))        
        if len(self.layer_inputs) == 0:
            raise RuntimeError('No input variables found for layers %s' % name)
        elif len(self.layer_inputs) == 1:
            layer_input = self.layer_inputs[0]
        else:
            layer_input = list(self.layer_inputs)            
            
        layer_output = op(self, layer_input, *args, **kwargs)
        
        self.layers[name] = layer_output
        self.feed(layer_output)
        
        return self
    return layer_decorated
    
    
class Model(object):
    def __init__(self, trainable=True):
        self.layers = dict()      
        self.trainable = trainable
        
        self.layer_inputs = []        
        self.setup()
    
    
    def build_loss(self):
        raise NotImplementedError('Must be subclassed.')
    
    
    def setup(self):        
        raise NotImplementedError('Must be subclassed.')
     
    
    @layer
    def embed(self, layer_input, vocabulary_size, embedding_size, name, trainable=True):
        with tf.variable_scope(name) as scope:
            init_embedding = tf.random_uniform_initializer(-1.0, 1.0)
            embeddings = self.make_var('weights', [vocabulary_size, embedding_size], init_embedding, None, trainable)
            shape = tf.shape(layer_input)
            
            reshaped_input = tf.reshape(layer_input, [-1])
            e = tf.nn.embedding_lookup(embeddings, reshaped_input)
            reshaped_e = tf.reshape(e, [shape[0], shape[1], shape[2], embedding_size])
            return reshaped_e
    
     
    @layer
    def sepconv(self, layer_input, k_h, k_w, cardinality, compression, name, activation='relu', trainable=True):
        """ customized seperable convolution
        """
        convolve = lambda input, filter: tf.nn.conv2d(input, filter, [1,1,1,1], 'SAME')
        activate = lambda z: tf.nn.relu(z, 'relu')
        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, 0.01)
            init_biases = tf.constant_initializer(0.0)
            regularizer = self.l2_regularizer(self.weight_decay)
            c_i = layer_input.get_shape().as_list()[-1]
            
            layer_output = []
            c = c_i / cardinality / compression
            for _ in range(cardinality):
                a = self.convolution(convolve, activate, layer_input, 1, 1, c_i, c,
                                     init_weights, init_biases, regularizer, trainable, '0_{}'.format(_))                
                a = self.convolution(convolve, activate, a, k_h, k_w, c, c, 
                                     init_weights, init_biases, regularizer, trainable, '1_{}'.format(_))
                a = self.convolution(convolve, activate, a, 1, 1, c, c_i, 
                                     init_weights, init_biases, regularizer, trainable, '2_{}'.format(_))
                layer_output.append(a)
            layer_output = tf.add_n(layer_output)
            return tf.add(layer_output, layer_input)
        
    
    @layer
    def up_sepconv(self, layer_input, k_h, k_w, cardinality, compression, name, activation='relu', trainable=True):
        """ customized upscale seperable convolution
        """
        convolve = lambda input, filter: tf.nn.conv2d(input, filter, [1,1,1,1], 'SAME')
        activate = lambda z: tf.nn.relu(z, 'relu')        
        with tf.variable_scope(name) as scope:
            shape = tf.shape(layer_input)
            h = shape[1]
            w = shape[2]
            layer_input = tf.image.resize_nearest_neighbor(layer_input, [2*h, 2*w])
            init_weights = tf.truncated_normal_initializer(0.0, 0.01)
            init_biases = tf.constant_initializer(0.0)
            regularizer = self.l2_regularizer(self.weight_decay)
            c_i = layer_input.get_shape().as_list()[-1]
            
            layer_output = []
            c = c_i / cardinality / compression
            for _ in range(cardinality):
                a = self.convolution(convolve, activate, layer_input, 1, 1, c_i, c,
                                     init_weights, init_biases, regularizer, trainable, '0_{}'.format(_))                
                a = self.convolution(convolve, activate, a, k_h, k_w, c, c, 
                                     init_weights, init_biases, regularizer, trainable, '1_{}'.format(_))
                a = self.convolution(convolve, activate, a, 1, 1, c, c_i, 
                                     init_weights, init_biases, regularizer, trainable, '2_{}'.format(_))
                layer_output.append(a)
            layer_output = tf.add_n(layer_output)
            return tf.add(layer_output, layer_input)
        
        
    @layer
    def dense_block(self, layer_input, k_h, k_w, c_o, depth, name, activation='relu', trainable=True):
        convolve = lambda input, filter: tf.nn.conv2d(input, filter, [1,1,1,1], 'SAME')
        activate = lambda z: tf.nn.relu(z, 'relu')
        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, 0.01)
            init_biases = tf.constant_initializer(0.0)
            regularizer = self.l2_regularizer(self.weight_decay)  
            
            layer_tmp = layer_input
            for d in range(depth):          
                c_i = layer_tmp.get_shape()[-1]
                a = self.convolution(convolve, activate, layer_tmp, 1, 1, c_i, c_i//2,
                                     init_weights, init_biases, regularizer, trainable)
                
                a = self.convolution(convolve, activate, a, k_h, k_w, c_i, c_o, 
                                     init_weights, init_biases, regularizer, trainable)
                
                layer_tmp = tf.concat([a, layer_input], 3)
                
            return layer_tmp
            
        
    @layer
    def conv(self, layer_input, k_h, k_w, c_o, s_h, s_w, name, activation='relu', trainable=True):
        convolve = lambda input, filter: tf.nn.conv2d(input, filter, [1,s_h,s_w,1], 'SAME')
        activate = lambda z: tf.nn.relu(z, 'relu')
        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, 0.01)
            init_biases = tf.constant_initializer(0.0)
            regularizer = self.l2_regularizer(self.weight_decay)
            c_i = layer_input.get_shape()[-1]
            
            a = self.convolution(convolve, activate, layer_input, k_h, k_w, c_i, c_o, 
                                 init_weights, init_biases, regularizer, trainable)
            return a  
    
    
    @layer
    def up_conv(self, layer_input, k_h, k_w, c_o, s_h, s_w, name, activation='relu', trainable=True):
        convolve = lambda input, filter: tf.nn.conv2d(input, filter, [1,s_h,s_w,1], 'SAME')
        activate = lambda z: tf.nn.relu(z, 'relu')        
        with tf.variable_scope(name) as scope:
            shape = tf.shape(layer_input)
            h = shape[1]
            w = shape[2]
            layer_input = tf.image.resize_nearest_neighbor(layer_input, [2*h, 2*w])
            init_weights = tf.truncated_normal_initializer(0.0, 0.01)
            init_biases = tf.constant_initializer(0.0)
            regularizer = self.l2_regularizer(self.weight_decay)
            c_i = layer_input.get_shape()[-1]
            
            a = self.convolution(convolve, activate, layer_input, k_h, k_w, c_i, c_o, 
                                 init_weights, init_biases, regularizer, trainable)
            return a  
    
    
    @layer
    def concat(self, layer_input, axis, name):
        return tf.concat(layer_input, axis)
    
    
    @layer
    def max_pool(self, layer_input, k_h, k_w, s_h, s_w, name, padding='VALID'):
        return tf.nn.max_pool(layer_input, [1,k_h,k_w,1], [1,s_h,s_w,1], name=name, padding=padding)
    
    
    @layer
    def softmax(self, layer_input, name):
        return tf.nn.softmax(layer_input, name=name)       
        
    
    def convolution(self, convolve, activate, input, k_h, k_w, c_i, c_o, init_weights, init_biases, regularizer, trainable, name=''):   
        kernel = self.make_var('weights'+name, [k_h, k_w, c_i, c_o], init_weights, regularizer, trainable) 
        biases = self.make_var('biases'+name, [c_o], init_biases, None, trainable)
        tf.summary.histogram('w', kernel)
        tf.summary.histogram('b', biases)
        wx = convolve(input, kernel)
        a = activate(tf.nn.bias_add(wx, biases))
        a = tf.contrib.layers.instance_norm(a, center=False, scale=False)
        return a
    
    
    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                factor = tf.convert_to_tensor(weight_decay, name='weight_decay')
                return tf.multiply(factor, tf.nn.l2_loss(tensor), name='decayed_value')
        return regularizer
    
    
    def make_var(self, name, shape, initializer=None, regularizer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, trainable=trainable)      
    
    
    def feed(self, *args):
        assert len(args) != 0
        
        self.layer_inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.layer_inputs.append(layer)
        return self
        
        
    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer
        
        
    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in list(self.layers.items())) + 1
        return '%s_%d' % (prefix, id)