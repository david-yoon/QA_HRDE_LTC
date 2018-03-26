#-*- coding: utf-8 -*-

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from scipy.stats import rankdata

"""
    desc  : True case : 2nd index
    
    inputs: 
        sess: tf session
        model: model for test
        data: such as the valid_set, test_set...
        N : N test case (1:n test)
    
    return:
        avg_ce: cross_entropy
        avg_accr : accuracy
"""
def run_test_r_order(sess, model, batch_gen, data, N):
    
    batch_ce = []
    
    batch_correct = []
    batch_correct_R2 = []
    batch_correct_R5 = []
    
    softmax, ce = 0, 0
    test_index = 0
    
    itr_loop = len(data) / model.batch_size
    
    for test_itr in xrange( itr_loop ):
        
        prob_labels = []
        batch_ce = []
        
        ## 1. probability for the the case (0: true, 1~9: false)
        
        rand_x = [i for i in xrange(10)]

        # random index
        
        tmp_x = rand_x[1:]
        shuffle(tmp_x)
        rand_x[1:] = tmp_x
        
        rand_x = rand_x[:N]
        
        
        for loop in xrange(N):

            # 9 -> 0  
            raw_encoder_inputs, raw_encoderR_inputs, raw_encoder_seq, raw_context_seq, raw_encoderR_seq, raw_target_label = batch_gen.get_batch(
                                                                            data=data,
                                                                            batch_size=model.batch_size,
                                                                            encoder_size=model.encoder_size,
                                                                            context_size = model.context_size,
                                                                            encoderR_size=model.encoderR_size,
                                                                            is_test=True,
                                                                            start_index= (test_itr* model.batch_size),
                                                                            target_index= rand_x[N -loop -1] )


            # prepare data which will be push from pc to placeholder
            input_feed = {}
            
            input_feed[model.encoder_inputs] = raw_encoder_inputs
            input_feed[model.encoderR_inputs] = raw_encoderR_inputs

            input_feed[model.encoder_seq_length] = raw_encoder_seq
            input_feed[model.context_seq_length] = raw_context_seq
            input_feed[model.encoderR_seq_length] = raw_encoderR_seq

            input_feed[model.y_label] = raw_target_label
            
            input_feed[model.dr_prob] = 1.0          # no drop out while evaluating
            input_feed[model.dr_prob_con] = 1.0   # no drop out while evaluating 
            input_feed[model.dr_memory_prob] = 1.0

            try:
                bprob, b_loss, lo = sess.run([model.batch_prob, model.batch_loss, model.loss], input_feed)
            except:
                print "excepetion occurs in valid step : " + str(test_itr)
                pass
            
            
            #if loop is (N-1):
            batch_ce.append( lo )
            
            prob_labels.append( np.reshape(bprob, model.batch_size) )
            
        prob_reshape = np.transpose(prob_labels).reshape(model.batch_size, len(prob_labels))
            
        rank = [ N - rankdata(x, method='max') for x in prob_reshape]
        
        batch_correct.append( sum([1 for x in rank if x[-1] < 1 ]) )
        
        if N == 10:
            batch_correct_R2.append( sum([1 for x in rank if x[-1] < 2]) )
            batch_correct_R5.append( sum([1 for x in rank if x[-1] < 5]) )
        

    avg_ce = sum(batch_ce) / ( len(data)/float(model.batch_size) ) / N
    
    avg_accr = 0
    avg_accr_R2 = 0
    avg_accr_R5 = 0
    
    avg_accr = sum(batch_correct) / float( itr_loop * model.batch_size )
    if N == 10:
        avg_accr_R2 = sum(batch_correct_R2) / float( itr_loop * model.batch_size ) 
        avg_accr_R5 = sum(batch_correct_R5) / float( itr_loop * model.batch_size )
        
        
    model.valid_loss = avg_ce
    model.accuracy = avg_accr
    
    value1 = summary_pb2.Summary.Value(tag="valid_loss_"+str(N), simple_value=avg_ce)
    value2 = summary_pb2.Summary.Value(tag="valid_accuracy_"+str(N), simple_value=avg_accr)
    summary = summary_pb2.Summary(value=[value1, value2])
    
    return avg_ce, avg_accr, summary, avg_accr_R2, avg_accr_R5