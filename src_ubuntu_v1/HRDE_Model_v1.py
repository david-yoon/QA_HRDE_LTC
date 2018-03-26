
#-*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper 

from tensorflow.core.framework import summary_pb2
from random import shuffle

from HRDE_process_data_v1 import *
from HRDE_evaluation import *

import os
import time
import argparse
from random import shuffle

class HRDualEncoderModel:
    
    
    def __init__(self, voca_size, batch_size,
                 encoder_size, context_size, encoderR_size, 
                 num_layer, hidden_dim,
                 num_layer_con, hidden_dim_con,
                 lr, embed_size, use_glove,
                 dr, dr_con,
                 memory_dim, topic_size):
        
        self.source_vocab_size = voca_size
        self.target_vocab_size = voca_size
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.context_size = context_size
        self.encoderR_size = encoderR_size
        self.num_layers = num_layer
        self.hidden_dim = hidden_dim
        self.num_layers_con = num_layer_con
        self.hidden_dim_con = hidden_dim_con
        self.lr = lr
        self.embed_size = embed_size
        self.use_glove = use_glove
        self.dr = dr
        self.dr_con = dr_con
        self.memory_dim = memory_dim
        self.topic_size = topic_size
    
        self.encoder_inputs = []
        self.context_inputs = []
        self.encoderR_inputs =[]
        self.y_label =[]

        self.M = None
        self.b = None
        
        self.y = None
        self.optimizer = None

        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None
        
        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        print '[launch] create placeholders'
        with tf.name_scope('data'):
            
            # [ batch X encoding_length, time_step (encoder_size) ]
            self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="encoder")
            # [ batch, time_step ] 
            self.encoderR_inputs = tf.placeholder(tf.int32, shape=[None, None], name="encoderR")
            
            
            # [ batch X encoding_length X time_step ]  
            self.encoder_seq_length = tf.placeholder(tf.int32, shape=[None], name="encoder_seq_len")
            # [ batch X encoding_length ]
            self.context_seq_length = tf.placeholder(tf.int32, shape=[None], name="context_seq_len")
            # [ batch X time_step ] 
            self.encoderR_seq_length = tf.placeholder(tf.int32, shape=[None], name="encoderR_seq_len")

            # [ batch, label ]
            self.y_label = tf.placeholder(tf.float32, shape=[None, None], name="label")
            
            self.dr_prob = tf.placeholder(tf.float32, name="dropout")
            self.dr_prob_con = tf.placeholder(tf.float32, name="dropout_con")
            self.dr_memory_prob = tf.placeholder(tf.float32, name="dropout_memory") # just for matching evaluation code with memory net version
        
            # for using pre-trained embedding
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.source_vocab_size, self.embed_size], name="embedding_placeholder")
        
        
    def _create_embedding(self):
        print '[launch] create embedding'
        with tf.name_scope('embed_layer'):
            self.embed_matrix = tf.Variable(tf.random_normal([self.source_vocab_size, self.embed_size],
                                                             mean=0.0,
                                                             stddev=0.01,
                                                             dtype=tf.float32,                                                             
                                                             seed=None),
                                            trainable = True,
                                            name='embed_matrix')
            
            self.embed_en       = tf.nn.embedding_lookup(self.embed_matrix, self.encoder_inputs, name='embed_encoder')
            self.embed_enR      = tf.nn.embedding_lookup(self.embed_matrix, self.encoderR_inputs, name='embed_encoderR')

            
    def _use_external_embedding(self):
        print '[launch] use pre-trained embedding'
        self.embedding_init = self.embed_matrix.assign(self.embedding_placeholder)
            
    # cell instance
    def gru_cell(self, hidden_dim):
        return tf.contrib.rnn.GRUCell(num_units=hidden_dim)
    
    # cell instance with drop-out wrapper applied
    def gru_drop_out_cell(self, hidden_dim, dr_prob):
        return tf.contrib.rnn.DropoutWrapper(self.gru_cell( hidden_dim ), input_keep_prob=dr_prob, output_keep_prob=dr_prob)                      
            
            
    def _create_gru_hrde_model(self):
        print '[launch] create gru model'
        
        with tf.name_scope('siaseme_RNN') as scope:
            # enoder RNN
            with tf.variable_scope("siaseme_GRU", reuse=False, initializer=tf.orthogonal_initializer()):
                single_cell_en = tf.contrib.rnn.GRUCell(self.hidden_dim)
                single_cell_en = tf.contrib.rnn.DropoutWrapper(single_cell_en, input_keep_prob=self.dr_prob, output_keep_prob=self.dr_prob)
                
                cells_en = tf.contrib.rnn.MultiRNNCell(  [ self.gru_drop_out_cell( self.hidden_dim, self.dr_prob )  for _ in range( self.num_layers ) ] )

                (outputs_en, self.states_en) = tf.nn.dynamic_rnn(
                                                  cell=cells_en,
                                                  inputs= self.embed_en,
                                                  dtype=tf.float32,
                                                  sequence_length=self.encoder_seq_length,
                                                  time_major=False)

            # response RNN
            with tf.variable_scope("siaseme_GRU", reuse=True, initializer=tf.orthogonal_initializer()):      
                single_cell_enR = tf.contrib.rnn.GRUCell(self.hidden_dim)
                single_cell_enR = tf.contrib.rnn.DropoutWrapper(single_cell_enR, input_keep_prob=self.dr_prob, output_keep_prob=self.dr_prob)
               
                cells_enR = tf.contrib.rnn.MultiRNNCell(  [ self.gru_drop_out_cell( self.hidden_dim, self.dr_prob )  for _ in range( self.num_layers ) ]  )

                (outputs_enR, self.states_enR) = tf.nn.dynamic_rnn(
                                                  cell=cells_enR,
                                                  inputs= self.embed_enR,
                                                  dtype=tf.float32,
                                                  sequence_length=self.encoderR_seq_length,
                                                  time_major=False)                
         
        
        with tf.name_scope('siaseme_Context_RNN') as scope:
        
            # context RNN for encoder
            with tf.variable_scope("Context_GRU", reuse=False, initializer=tf.orthogonal_initializer()):      
                # context RNN
                #with tf.variable_scope("context_RNN", reuse=False):
                single_cell_con = tf.contrib.rnn.GRUCell(self.hidden_dim_con)
                single_cell_con = tf.contrib.rnn.DropoutWrapper(single_cell_con, input_keep_prob=self.dr_prob_con, output_keep_prob=self.dr_prob_con)
                
                cells_con = tf.contrib.rnn.MultiRNNCell( [ self.gru_drop_out_cell( self.hidden_dim_con, self.dr_prob_con )  for _ in range( self.num_layers_con ) ] )

                # make data for context input
                con_input = tf.reshape( self.states_en[-1], [self.batch_size, self.context_size, self.hidden_dim])

                (outputs_con, last_states_con) = tf.nn.dynamic_rnn(
                    cell=cells_con,
                    inputs= con_input,
                    dtype=tf.float32,
                    sequence_length=self.context_seq_length,
                    time_major=False)
                
                
                self.final_encoder = last_states_con[-1]
                

            # context RNN for response
            with tf.variable_scope("Context_GRU", reuse=True, initializer=tf.orthogonal_initializer()):      
                # context RNN
                #with tf.variable_scope("context_RNN", reuse=False):
                single_cell_conR = tf.contrib.rnn.GRUCell(self.hidden_dim_con)
                single_cell_conR = tf.contrib.rnn.DropoutWrapper(single_cell_conR, input_keep_prob=self.dr_prob_con, output_keep_prob=self.dr_prob_con)

                cells_conR = tf.contrib.rnn.MultiRNNCell( [ self.gru_drop_out_cell( self.hidden_dim_con, self.dr_prob_con )  for _ in range( self.num_layers_con ) ] )

                # make data for context input
                con_inputR = tf.reshape( self.states_enR[-1], [self.batch_size, 1, self.hidden_dim])

                (outputs_conR, last_states_conR) = tf.nn.dynamic_rnn(
                    cell=cells_conR,
                    inputs= con_inputR,
                    dtype=tf.float32,
                    sequence_length=np.ones(self.batch_size, dtype=int).tolist(),
                    time_major=False)
                
                self.final_encoderR  = last_states_conR[-1]
                
            self.final_encoder_dimension   = self.hidden_dim_con
            self.final_encoderR_dimension = self.hidden_dim_con

            
    def _add_memory_network(self):
        print '[launch] add memory network'
        
        with tf.name_scope('memory_network') as scope:
    
            # memory space for latent topic
            self.memory = tf.Variable(tf.random_uniform( [self.memory_dim, self.topic_size],
                                                       minval= -0.25,
                                                       maxval= 0.25,
                                                       dtype=tf.float32,
                                                       seed=None),
                                                       name="latent_topic_memory")

            
            self.memory_W = tf.Variable(tf.random_uniform( [self.hidden_dim, self.memory_dim],
                                                       minval= -0.25,
                                                       maxval= 0.25,
                                                       dtype=tf.float32,
                                                       seed=None),
                                                       name="latent_topic_memory")
            
            topic_sim_project = tf.matmul( self.final_encoder, self.memory_W )
            
            topic_sim = tf.matmul( topic_sim_project, self.memory )

            # normalize
            topic_sim_norm = tf.nn.softmax( logits=topic_sim, dim=-1)

            shaped_input = tf.reshape( topic_sim_norm, [self.batch_size, self.topic_size])
            topic_sim_mul_memory = tf.scan( lambda a, x : tf.multiply( self.memory, x ), shaped_input, initializer=self.memory)
            rsum = tf.reduce_sum( topic_sim_mul_memory, axis=-1)

            # final context 
            self.final_encoder = tf.concat( [self.final_encoder, rsum], axis=-1 )
            
        self.final_encoder_dimension   = self.hidden_dim_con + self.memory_dim   # concat 으로 늘어났음
        self.final_encoderR_dimension  = self.hidden_dim_con
            
            
    def _create_output_layers(self):
        print '[launch] create output projection layer'        
           
        with tf.name_scope('output_layer') as scope:
            
            self.M = tf.Variable(tf.random_uniform([self.final_encoder_dimension, self.final_encoderR_dimension],
                                                   minval= -0.25,
                                                   maxval= 0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                                   name="similarity_matrix")
            
            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32), name="output_bias")
            
            
            # (c * M) * r + b
            tmp_y = tf.matmul(self.final_encoder, self.M)
            batch_y_hat = tf.reduce_sum( tf.multiply(tmp_y, self.final_encoderR), 1, keep_dims=True ) + self.b
            self.batch_prob = tf.sigmoid( batch_y_hat )
                
        
        with tf.name_scope('loss') as scope:
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=batch_y_hat, labels=self.y_label )
            self.loss = tf.reduce_mean( self.batch_loss  )
    
    
    def _create_optimizer(self):
        print '[launch] create optimizer'
        
        with tf.name_scope('optimizer') as scope:
            
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(t=grad, clip_value_min=-10, clip_value_max=10), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)
    
    
    def _create_summary(self):
        print '[launch] create summary'
        
        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    
    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        
        if self.use_glove == 1:
            self._use_external_embedding()
            
        self._create_gru_hrde_model()
        
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()

        
# for training         
def train_step(sess, model, batch_gen):
    raw_encoder_inputs, raw_encoderR_inputs, raw_encoder_seq, raw_context_seq, raw_encoderR_seq, raw_target_label = batch_gen.get_batch(
                                        data=batch_gen.train_set,
                                        batch_size=model.batch_size,
                                        encoder_size=model.encoder_size,
                                        context_size=model.context_size,
                                        encoderR_size=model.encoderR_size,
                                        is_test=False
                                        )

    # prepare data which will be push from pc to placeholder
    input_feed = {}
    
    input_feed[model.encoder_inputs] = raw_encoder_inputs
    input_feed[model.encoderR_inputs] = raw_encoderR_inputs

    input_feed[model.encoder_seq_length] = raw_encoder_seq
    input_feed[model.context_seq_length] = raw_context_seq
    input_feed[model.encoderR_seq_length] = raw_encoderR_seq

    input_feed[model.y_label] = raw_target_label
    input_feed[model.dr_prob] = model.dr
    input_feed[model.dr_prob_con] = model.dr_con

    
    _, summary = sess.run([model.optimizer, model.summary_op], input_feed)
    
    return summary

    
def train_model(model, batch_gen, num_train_steps, valid_freq, is_save=0, graph_dir_name='default'):
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    summary = None
    val_summary = None
    
    CAL_ACCURACY_FROM = 1
    MAX_EARLY_STOP_COUNT = 7
    
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
        early_stop_count = MAX_EARLY_STOP_COUNT
        
        if model.use_glove == 1:
            sess.run(model.embedding_init, feed_dict={ model.embedding_placeholder: batch_gen.get_glove() })
            print 'use pre-trained embedding vector'

        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('save/' + graph_dir_name + '/'))
        if ckpt and ckpt.model_checkpoint_path:
            print ('from check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)
            

        writer = tf.summary.FileWriter('./graph/'+graph_dir_name, sess.graph)

        initial_time = time.time()
        
        max_accr = 0
        
        for index in xrange(num_train_steps):

            try:
                # run train 
                summary = train_step(sess, model, batch_gen)
                writer.add_summary( summary, global_step=model.global_step.eval() )
                
            except:
                print "excepetion occurs in train step"
                pass
                
            
            # run validation
            if (index + 1) % valid_freq == 0:
                
                ce, accr, val_summary, _, _ = run_test_r_order(sess=sess, model=model, batch_gen=batch_gen,
                                             data=batch_gen.valid_set, N=2)
                
                writer.add_summary( val_summary, global_step=model.global_step.eval() )
                
                end_time = time.time()

                if index > CAL_ACCURACY_FROM:

                    if ( accr > max_accr ):
                        max_accr = accr

                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )

                        early_stop_count = MAX_EARLY_STOP_COUNT
                        
                        _, test_accr, _, _, _ = run_test_r_order(sess=sess, model=model, batch_gen=batch_gen,
                                             data=batch_gen.test_set, N=2)
                        
                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print "early stopped"
                            break
                             
                        test_accr = 0
                        early_stop_count = early_stop_count -1

                    print str( int(end_time - initial_time)/60 ) + " mins" + \
                        "\t step/seen/itr: " + str( model.global_step.eval() ) + "/ " + \
                                               str( model.global_step.eval() * model.batch_size ) + "/" + \
                                               str( round( model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)  ) + \
                        "\tval: " + str( round(accr, 4) )  + "\tt: " + str( round(test_accr, 4) )
                            
                
        writer.close()
            
        print ('Total steps : {}'.format(model.global_step.eval()) )


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
        
def main(batch_size, encoder_size, context_size, encoderR_size, num_layer, hidden_dim, num_layer_con, hidden_dim_con,
         embed_size, num_train_steps, lr, valid_freq, is_save, graph_dir_name, is_test, use_glove, dr, dr_con,
         memory_dim, topic_size):
    
    if is_save is 1:
        create_dir('save/')
        create_dir('save/'+ graph_dir_name )
    
    create_dir('graph/')
    create_dir('graph/' + graph_dir_name )
    
    batch_gen = ProcessData(is_test=is_test)
    if is_test == 1:
        valid_freq = 100
    
    model = HRDualEncoderModel(voca_size=len(batch_gen.voca),
                               batch_size=batch_size,
                               encoder_size=encoder_size,
                               context_size=context_size,
                               encoderR_size=encoderR_size,
                               num_layer=num_layer,                 
                               hidden_dim=hidden_dim,
                               num_layer_con=num_layer_con,
                               hidden_dim_con=hidden_dim_con,
                               lr=lr,
                               embed_size=embed_size,
                               use_glove = use_glove,
                               dr=dr,
                               dr_con=dr_con,
                               memory_dim = memory_dim,
                               topic_size=topic_size
                               )
    
    model.build_graph()
    
    train_model(model, batch_gen, num_train_steps, valid_freq, is_save, graph_dir_name)
    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--encoder_size', type=int, default=80)
    p.add_argument('--context_size', type=int, default=10)
    p.add_argument('--encoderR_size', type=int, default=80)
    
    # siaseme RNN
    p.add_argument('--num_layer', type=int, default=2)
    p.add_argument('--hidden_dim', type=int, default=300)
    
    # context RNN
    p.add_argument('--num_layer_con', type=int, default=2)
    p.add_argument('--hidden_dim_con', type=int, default=300)
    
    p.add_argument('--embed_size', type=int, default=200)
    p.add_argument('--num_train_steps', type=int, default=10000)
    p.add_argument('--lr', type=float, default=1e-1)
    p.add_argument('--valid_freq', type=int, default=500)
    p.add_argument('--is_save', type=int, default=0)
    p.add_argument('--graph_prefix', type=str, default="default")
    
    p.add_argument('--is_test', type=int, default=0)
    p.add_argument('--use_glove', type=int, default=0)
    
    p.add_argument('--dr', type=float, default=1.0)
    p.add_argument('--dr_con', type=float, default=1.0)
    
    # latent topic
    p.add_argument('--memory_dim', type=int, default=32)
    p.add_argument('--topic_size', type=int, default=0)
    
    args = p.parse_args()
    
    graph_name = args.graph_prefix + \
                    '_b' + str(args.batch_size) + \
                    '_es' + str(args.encoder_size) + \
                    '_eRs' + str(args.encoderR_size) + \
                    '_cs' + str(args.context_size) + \
                    '_L' + str(args.num_layer) + \
                    '_H' + str(args.hidden_dim) + \
                    '_Lc' + str(args.num_layer_con) + \
                    '_Hc' + str(args.hidden_dim_con) + \
                    '_G' + str(args.use_glove) + \
                    '_dr' + str(args.dr)  + \
                    '_drc' + str(args.dr_con) + \
                    '_M' + str(args.memory_dim) + \
                    '_T' + str(args.topic_size)
    
    main(
        batch_size=args.batch_size,
        encoder_size=args.encoder_size,
        context_size=args.context_size,
        encoderR_size=args.encoderR_size,
        num_layer=args.num_layer,
        hidden_dim=args.hidden_dim,
        num_layer_con=args.num_layer_con,
        hidden_dim_con=args.hidden_dim_con,
        embed_size=args.embed_size, 
        num_train_steps=args.num_train_steps,
        lr=args.lr,
        valid_freq=args.valid_freq,
        is_save=args.is_save,
        graph_dir_name=graph_name,
        is_test=args.is_test,
        use_glove=args.use_glove,
        dr=args.dr,
        dr_con=args.dr_con,
        memory_dim=args.memory_dim,
        topic_size=args.topic_size
        )
