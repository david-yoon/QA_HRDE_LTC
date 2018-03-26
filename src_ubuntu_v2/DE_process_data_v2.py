# coding: utf-8

import numpy as np
import random
import pickle


"""
Train data : 0: question,  1: answer,  2: label (1/0)
Valid data : 0: question,  1: true,  [2:10] : false
Test  data : 0: question,  1: true,  [2:10] : false

train_set :  0: question,  1: answer,  2: lable (1/0)
valid_set :  0: question,  1: answer ( [0:9] list , 0 is answer )
test_set  :  0: question,  1: answer ( [0:9] list , 0 is answer )

"""

DATA_DIR = '../data/ubuntu_v2/'

DATA_FILE_NAME = 'v2_data.pkl'
TEST_DATA_FILE_NAME = 'v2_data_test.pkl'  # contains only 500 samples

VOCA_FILE_NAME = 'v2_dic.pkl'
GLOVE_FILE_NAME = 'v2_glove.pkl'


class ProcessData:
    
    def __init__(self, is_test):
        
        print 'IS_TEST = ' + str(is_test)
        
        self.is_test = is_test
        self.voca = None
        self.pad_index = 0
        self.index2word = {}
        
        self.train_set = []
        self.valid_set = []
        self.test_set = []
        
        self.load_data()
        self.create_train_set()
        self.create_valid_set()
        self.create_test_set()
        
        
        
    def load_data(self):
        
        if self.is_test:
            print 'load data : ' + TEST_DATA_FILE_NAME
            self.train_data, self.valid_data, self.test_data = pickle.load(open(DATA_DIR + TEST_DATA_FILE_NAME, 'r'))
        else:
            self.train_data, self.valid_data, self.test_data = pickle.load(open(DATA_DIR + DATA_FILE_NAME, 'r'))
            print 'load data : ' + DATA_FILE_NAME
        
        self.voca = pickle.load(open(DATA_DIR + VOCA_FILE_NAME, 'r') )
        self.W_glove_init = pickle.load(open(DATA_DIR + GLOVE_FILE_NAME, 'r') )
        
        self.pad_index = self.voca['_PAD_']
        #self.pad_index = 0
        
        for w in self.voca:
            self.index2word[self.voca[w]] = w
        
        print '[completed] load data'
        print 'voca size (include _PAD_, _UNK_): ' + str( len(self.voca) )
        

    # convert to soucre, target, label
    def create_train_set(self):
        
        data_len = len(self.train_data['c'])
        
        for index in xrange(data_len):
            
            source_ids = self.train_data['c'][index]
            target_ids = self.train_data['r'][index]
            label = float(self.train_data['y'][index])
            
            self.train_set.append( [source_ids, target_ids, label] )
        
        print '[completed] create trian set : ' + str(len(self.train_set))
            
      
    # create valid set : response 0 : true, response 1-9 : false
    def create_valid_set(self):
        
        data_len = len(self.valid_data['c'])
            
        for index in xrange(data_len / 10):
        
            t_index = index * 10
            
            source_ids = self.valid_data['c'][t_index]

            list_target_ids = []
            for i in xrange(10) :
                list_target_ids.append( self.valid_data['r'][t_index+i] )

            self.valid_set.append( [source_ids, list_target_ids] )
    
        print '[completed] create valid set : ' + str(len(self.valid_set))
        
        
    # create test set : test_data (word) -> test_set (index number)        
    def create_test_set(self):
        
        data_len = len(self.test_data['c'])
            
        for index in xrange(data_len / 10):
        
            t_index = index * 10
            
            source_ids = self.test_data['c'][t_index]

            list_target_ids = []
            for i in xrange(10) :
                list_target_ids.append( self.test_data['r'][t_index+i] )

            self.test_set.append( [source_ids, list_target_ids] )
            
        print '[completed] create test set : ' + str(len(self.test_set))
            

    def get_glove(self):
        
        print 'from glove initial'
        return self.W_glove_init
    
    
    """
        inputs: 
            data: 
            batch_size : 
            encoder_size : max encoder time step
            context_size : max context encoding time step
            encoderR_size : max decoder time step
            
            is_test : test case batch data generation
            start_index : start index
            target_index : 0 : true
                          1-9: false

        return:
            encoder_inputs : [batch x context_size, time_step] 
            encoderR_inputs : [batch, time_step] 
            encoder_seq : 
            context_seq  : 
            encoderR_seq : 
            target_labels : label
    """
    def get_batch(self, data, batch_size, encoder_size, encoderR_size, is_test, start_index=0, target_index=1):

        encoder_inputs, encoderR_inputs, encoder_seq, encoderR_seq, target_labels = [], [], [], [], []
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in xrange(batch_size):

            if is_test is False:
                # train random sampling
                encoder_input, encoderR_input, target_label = random.choice(data)
            else:
                encoder_input = data[index][0] 
                encoderR_input = data[index][1][target_index]
                index = index +1
    
            
            # encoder inputs are padded
            encoder_pad = [self.pad_index] * (encoder_size - len(encoder_input))
            encoder_inputs.append( (encoder_input + encoder_pad)[:encoder_size]) 

            encoder_seq.append( min(len(encoder_input), encoder_size) )
            
            
            # encoderR inputs are padded
            encoderR_pad = [self.pad_index] * (encoderR_size - len(encoderR_input))
            encoderR_inputs.append( (encoderR_input + encoderR_pad)[:encoderR_size]) 

            encoderR_seq.append( min(len(encoderR_input), encoderR_size) )

            # Target Label for batch
            if is_test is False:
                target_labels.append( int(target_label) )
            else:
                if target_index is 0:
                    target_labels.append( int(1) )
                else:
                    target_labels.append( int(0) )

                    
        return encoder_inputs, encoderR_inputs, encoder_seq, encoderR_seq, np.reshape(target_labels, (batch_size, 1))