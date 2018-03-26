'''
evaluation RDE
'''

import os
import numpy as np

path = './save/'

list_dir = os.listdir(path)
list_de = []
list_de_mem = []
list_hrde = []
list_hrde_mem = []

for name in list_dir:
    if name[:7] == 'RDE_LTC':
        list_de_mem.append(name)
    elif name[:3] == 'RDE' :
        list_de.append(name)
    elif name[:8] == 'HRDE_LTC' :
        list_hrde_mem.append(name)
    elif name[:4] == 'HRDE' :
        list_hrde.append(name)

def run_single_eval(model, batch_gen, target_dir, N):
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    summary = None

    target_dir = target_dir + '/'
    print "target_dir:" + target_dir
    
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(target_dir ))
        if ckpt and ckpt.model_checkpoint_path:
            print ('from check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)

        ce, accr, summary, r2, r5 = run_test_r_order(sess=sess, model=model, batch_gen=batch_gen,
                                                 data=batch_gen.test_set, N=N)

    return accr, r2, r5



def evals(result_file_name, path, list_save, model, batch_gen):
  
    log_f = path + result_file_name
    print log_f

    with open(log_f, 'w') as f:
        
        n10_r1 = []
        n10_r2 = []
        n10_r5 = []
        n2_r1 = []

        for save_name in list_save:

            f.write(save_name + '\n')

            accr, r2, r5 = run_single_eval(model, batch_gen, path + save_name, 2)
            print accr, r2, r5
            n2_r1.append(accr)
            f.write( str(accr) + '\t')

            
            accr, r2, r5 = run_single_eval(model, batch_gen, path + save_name, 2)
            print accr, r2, r5
            n2_r1.append(accr)
            f.write( str(accr) + '\t')
            
            
            accr, r2, r5 = run_single_eval(model, batch_gen, path + save_name, 10)
            print accr, r2, r5
            n10_r1.append(accr)
            n10_r2.append(r2)
            n10_r5.append(r5)
            f.write( str(accr) + '\t' + str(r2) + '\t' + str(r5) + '\n')
            
        print "mean-std"
        print str(np.mean(n2_r1)), str(np.mean(n10_r1)), str(np.mean(n10_r2)), str(np.mean(n10_r5))
        print str( np.std(n2_r1)), str( np.std(n10_r1)), str( np.std(n10_r2)), str( np.std(n10_r5))
        
        f.write( str(np.mean(n2_r1)) + '\t' + str(np.mean(n10_r1)) + '\t' + str(np.mean(n10_r2)) + '\t' + str(np.mean(n10_r5)) + '\n')
        f.write( str( np.std(n2_r1)) + '\t' + str( np.std(n10_r1)) + '\t' + str( np.std(n10_r2)) + '\t' + str( np.std(n10_r5)) + '\n\n')
        
        
'''
RDE MODEL
'''
from DE_Model_v2 import *
from DE_process_data_v2 import *
from DE_evaluation import *
batch_gen = ProcessData(is_test=False)
#batch_gen = ProcessData(is_test=True)


voca_size = len(batch_gen.voca)
source_vocab_size = voca_size
target_vocab_size = voca_size

batch_size = 1024
encoder_size, encoderR_size = 160, 160

num_layer = 1
hidden_dim = 300

embed_size = 300
lr = 1e-3

use_glove = 1
dr = 1.0

model = DualEncoderModel(
            voca_size=len(batch_gen.voca),
            batch_size=batch_size,
            encoder_size=encoder_size,
            encoderR_size=encoderR_size,
            num_layer=num_layer,
            hidden_dim=hidden_dim,
            lr = lr,
            embed_size=embed_size,
            use_glove = use_glove,
            dr=dr,
            memory_dim=0,
            topic_size=0
            )

model.build_graph()


evals(result_file_name='result_RDE.txt', 
      path=path, 
      list_save=list_de, 
      model=model, 
      batch_gen=batch_gen)