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
parser.add_argument('--doc_path', type=str, default='data/taxi_small') 
parser.add_argument('--save_prefix', type=str, default='taxi', help='prefix for ckpt') # TBD: save log/models with prefix

# dict
parser.add_argument('--dict_path', type=str, default='dict/---') # not used if load_dict is True
parser.add_argument('--load_dict', type=bool, default=True, help='True to work based on an existing dict') 
parser.add_argument('--load_dict_from_path', type=str, default='dict/40000') 
parser.add_argument('--large_dict', type=bool, default=True, help='True to use large dict for future ext') 

# ckpt
parser.add_argument('--restore_ckpt', type=bool, default=True) 
parser.add_argument('--restore_embedding_only', type=bool, default=True) 
parser.add_argument('--embedding_file', type=str, default='../graph/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt')  

# training
parser.add_argument('--embedding_size', type=int, default=768) 
params = parser.parse_args()

class BertEmbedding(object):
    def __init__(self, vocab_size=119547, hidden_size=768, initializer_range=0.02, use_one_hot_embeddings=False, trainable=False):
        self.trainable = trainable    
        self.word_embedding_name = "word_embeddings"
        self.input_ids = tf.placeholder(tf.int32, shape=[None, None, None, 1], name='grid_table')

        with tf.variable_scope("bert"):
          with tf.variable_scope("embeddings"):
            # Perform embedding lookup on the word ids.
            (self.embedding_output, self.embedding_table) = self.embedding_lookup(
                input_ids=self.input_ids,
                vocab_size=vocab_size,
                embedding_size=hidden_size,
                initializer_range=initializer_range,
                word_embedding_name=self.word_embedding_name,
                use_one_hot_embeddings=use_one_hot_embeddings)
        
    def embedding_lookup(self, input_ids,
                         vocab_size,
                         embedding_size=128,
                         initializer_range=0.02,
                         word_embedding_name="word_embeddings",
                         use_one_hot_embeddings=False):
        """Looks up words embeddings for id tensor.
        
        Args:
          input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
            ids.
          vocab_size: int. Size of the embedding vocabulary.
          embedding_size: int. Width of the word embeddings.
          initializer_range: float. Embedding initialization range.
          word_embedding_name: string. Name of the embedding table.
          use_one_hot_embeddings: bool. If True, use one-hot method for word
            embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
            for TPUs.
        
        Returns:
          float Tensor of shape [batch_size, seq_length, embedding_size].
        """
        # This function assumes that the input is of shape [batch_size, seq_length,
        # num_inputs].
        #
        # If the input is a 2D tensor of shape [batch_size, seq_length], we
        # reshape to [batch_size, seq_length, 1].
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])
        
        embedding_table = tf.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=self.create_initializer(initializer_range),
            trainable=self.trainable)
        
        if use_one_hot_embeddings:
            flat_input_ids = tf.reshape(input_ids, [-1])
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
            output = tf.matmul(one_hot_input_ids, embedding_table)
        else:
            output = tf.nn.embedding_lookup(embedding_table, input_ids)
        
        input_shape = self.get_shape_list(input_ids)
        
        output = tf.reshape(output,
                            input_shape[0:-1] + [input_shape[-1] * embedding_size])
        return (output, embedding_table)

    def create_initializer(self, initializer_range=0.02):
        """Creates a `truncated_normal_initializer` with the given range."""
        return tf.truncated_normal_initializer(stddev=initializer_range)
    
    def get_shape_list(self, tensor, expected_rank=None, name=None):
        """Returns a list of the shape of tensor, preferring static dimensions.
        
        Args:
          tensor: A tf.Tensor object to find the shape of.
          expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
          name: Optional name of the tensor for the error message.
        
        Returns:
          A list of dimensions of the shape of tensor. All static dimensions will
          be returned as python integers, and dynamic dimensions will be returned
          as tf.Tensor scalars.
        """
        if name is None:
          name = tensor.name
        
        if expected_rank is not None:
          assert_rank(tensor, expected_rank, name)
        
        shape = tensor.shape.as_list()
        
        non_static_indexes = []
        for (index, dim) in enumerate(shape):
          if dim is None:
            non_static_indexes.append(index)
        
        if not non_static_indexes:
          return shape
        
        dyn_shape = tf.shape(tensor)
        for index in non_static_indexes:
          shape[index] = dyn_shape[index]
        return shape

if __name__ == '__main__':
    # data
    data_loader = DataLoader(params, for_train=True, load_dictionary=params.load_dict, data_split=0.75)

    # model
    bert = BertEmbedding() 
    
    load_variable = {"bert/embeddings/word_embeddings": bert.embedding_table}
    ckpt_saver = tf.train.Saver(load_variable, max_to_keep=50)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        if False:
            pass
        elif params.restore_embedding_only:
            try:
                ckpt = tf.train.get_checkpoint_state(params.embedding_file)
                print('Restoring from {}...'.format(ckpt_path))
                ckpt_saver.restore(sess, ckpt_path)
                print('Restored from {}'.format(ckpt_path))
            except:
                raise('Check your path {:s}'.format(ckpt_path))
            
        data = data_loader.next_batch()
        # one step training
        feed_dict = {
            bert.input_ids: data['grid_table'],
        }
        fetches = [bert.embedding_output, bert.embedding_table]
        (output, table) = sess.run(fetches=fetches, feed_dict=feed_dict)
        print(output[0][0][0])
    