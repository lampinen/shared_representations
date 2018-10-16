import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import block_diag
from orthogonal_matrices import random_orthogonal

######Parameters###################
init_eta = 0.002
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 10000
nruns = 10
num_inputs_per = 20
num_outputs_per = 30
num_hidden = 2
rank = 1
#nhidden = 6
#rseed = 2  #reproducibility
###################################



#x_data = {}
#y_data = {}
#x_data[1] = np.array([[1,0],[0,1],[0,0],[0,0] ])
#this_y_data = np.array(map(lambda x: [1*(x[0] or x[1]),x[0],x[1]],x_data[1]))
#y_data[1] = this_y_data.reshape([len(x_data[1]),3])
#x_data[2] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] ])
#this_y_data = np.array(map(lambda x: [1*(x[0] or x[1]),x[0],x[1],1*(x[2] or x[3]),x[2],x[3]],x_data[2]))
#y_data[2] = this_y_data.reshape([len(x_data[2]),6])
#x_data[3] = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
#this_y_data = np.array(map(lambda x: [1*(x[0] or x[1]),x[0],x[1],1*(x[2] or x[3]),x[2],x[3],1*(x[4] or x[5]),x[4],x[5]],x_data[3]))
#y_data[3] = this_y_data.reshape([len(x_data[3]),9])
#print "x_data:"
#print x_data
#print
#
#print "y_data:"
#print y_data
#print
#

def SVD_dataset(num_examples, num_outputs, num_nonempty=4, singular_value_multiplier=5):
    """Like the shared input modes dataset, but only one domain"""
    input_modes = random_orthogonal(num_examples)
    strengths = np.zeros(num_examples)
    strengths[:num_nonempty] = singular_value_multiplier * (np.array(range(num_nonempty, 0, -1)))

    def _strengths_to_S(strengths, num_outputs=num_outputs):
        if num_outputs > num_examples:
            return np.concatenate((np.diag(strengths), np.zeros((num_outputs-len(strengths), num_examples))), axis=0)
        else:
            return np.diag(strengths)[:num_outputs, :]

    S = _strengths_to_S(strengths)

    y_data = np.transpose(np.matmul(random_orthogonal(num_outputs), np.matmul(S, input_modes)))
    x_data = np.eye(len(y_data))
    return x_data, y_data, input_modes


for rseed in xrange(0, nruns):
    for network in ['nonlinear', 'linear']:
        for nlayer in [3,2]:
            for ndomains in [1,2]: #,3]:
                ninput = num_inputs_per*ndomains
                noutput = num_outputs_per*ndomains 
                nhidden = num_hidden
                alignment_options = ["aligned", "random"] if ndomains >1  else ["NA"]
                for alignment in alignment_options:
                    print "nlayer %i ndomains %i alignment %s run %i" % (nlayer, ndomains, alignment, rseed)
                    filename_prefix = "results/task_comp/%s_nlayer_%i_ndomains_%i_aligned_%s_rseed_%i_" %(network,nlayer,ndomains,alignment,rseed)

                    np.random.seed(rseed)
                    tf.set_random_seed(rseed)
                    this_x_data, this_y_data, _ = SVD_dataset(num_inputs_per, num_outputs_per, num_nonempty=rank)
                    if ndomains == 2:
                        if alignment == "random": 
                            this_x_data_2, this_y_data_2, _ = SVD_dataset(num_inputs_per, num_outputs_per, num_nonempty=rank)
                        else:
                            this_x_data_2, this_y_data_2 = this_x_data, this_y_data 

                        
                        x_data = block_diag(this_x_data, this_x_data_2)
                        y_data = block_diag(this_y_data, this_y_data_2)
                    else:
                        x_data = this_x_data
                        y_data = this_y_data
                    y_data = np.clip(y_data, 0, None)# + 0.1*np.clip(y_data, None, 0) # make the task leaky_relu-ed linear

                    input_ph = tf.placeholder(tf.float32, shape=[None, ninput])
                    target_ph = tf.placeholder(tf.float32, shape=[None, noutput])
                    if nlayer == 2:
                        W1 = tf.Variable(tf.random_uniform([nhidden,ninput],0.,2.0/(num_hidden+num_inputs_per)))
                        W2 = tf.Variable(tf.random_uniform([noutput,nhidden],0.,2./(num_hidden+num_outputs_per)))
                        internal_rep = tf.matmul(W1,tf.transpose(input_ph))
                        pre_output = tf.matmul(W2,internal_rep)
                    elif nlayer == 3:
                        W1 = tf.Variable(tf.random_uniform([nhidden,ninput],0.,2./(num_hidden+num_inputs_per)))
                        W2 = tf.Variable(tf.random_uniform([nhidden,nhidden],0.,1./num_hidden))
                        W3 = tf.Variable(tf.random_uniform([noutput,nhidden],0.,2./(num_hidden+num_outputs_per)))
                        internal_rep = tf.matmul(W1,tf.transpose(input_ph))
                            
                        pre_output = tf.matmul(W3,tf.matmul(W2,internal_rep))
                    elif nlayer == 4:
                        W1 = tf.Variable(tf.random_uniform([nhidden,ninput],0.,2./(num_hidden+num_inputs_per)))
                        W2 = tf.Variable(tf.random_uniform([nhidden,nhidden],0.,1./num_hidden))
                        W3 = tf.Variable(tf.random_uniform([nhidden,nhidden],0.,1./num_hidden))
                        W4 = tf.Variable(tf.random_uniform([noutput,nhidden],0.,2./(num_hidden+num_outputs_per)))
                        internal_rep = tf.matmul(W1,tf.transpose(input_ph))
                            
                        pre_output = tf.matmul(W4,tf.matmul(W3,tf.matmul(W2,internal_rep)))
                    else:
                        print "Error, invalid number of layers given"
                        exit(1)

                    output = tf.nn.leaky_relu(pre_output, alpha=0.1)
                    rep_mean_ph =  tf.placeholder(tf.float32, shape=[nhidden,1])

                    target_t = tf.transpose(target_ph)
                    loss = tf.reduce_sum(tf.square(output - target_t))# +0.05*(tf.nn.l2_loss(internal_rep))
                    d1_loss = tf.reduce_sum(tf.square(output[:num_outputs_per, :] - target_t[:num_outputs_per, :]))
                    linearized_loss = tf.reduce_sum(tf.square(pre_output - target_t))# +0.05*(tf.nn.l2_loss(internal_rep))
                    d1_linearized_loss = tf.reduce_sum(tf.square(pre_output[:num_outputs_per, :] - target_t[:num_outputs_per, :]))
                    output_grad = tf.gradients(loss,[output])[0]
                    W1_grad = tf.gradients(loss,[W1])[0]
                    eta_ph = tf.placeholder(tf.float32)
                    optimizer = tf.train.GradientDescentOptimizer(eta_ph)
                    train = optimizer.minimize(loss)
                    linearized_train = optimizer.minimize(linearized_loss)

                    init = tf.global_variables_initializer()

                    sess = tf.Session()
                    sess.run(init)

                    def test_accuracy():
                        MSE = sess.run(d1_linearized_loss if network == "linear" else d1_loss,
                                       feed_dict={input_ph: x_data[:num_inputs_per, :],target_ph: y_data[:num_inputs_per, :]})
                        MSE /= num_inputs_per 
                        return MSE

                    def print_outputs():
                        print sess.run(output,feed_dict={input_ph: x_data})


                    def print_preoutputs():
                        print sess.run(pre_output,feed_dict={input_ph: x_data})


                    def save_activations(tf_object,filename,remove_old=True):
                        if remove_old and os.path.exists(filename):
                            os.remove(filename)
                        with open(filename,'ab') as fout:
                            res = sess.run(tf_object, feed_dict={input_ph: x_data})
                            np.savetxt(fout, res, delimiter=',')


                    def save_weights(tf_object,filename,remove_old=True):
                        if remove_old and os.path.exists(filename):
                            os.remove(filename)
                        with open(filename,'ab') as fout:
                            np.savetxt(fout,sess.run(tf_object),delimiter=',')


                    def display_rep_similarity():
                        raise NotImplementedError("Needs updating")
                        reps = []
                        nsamples = len(x_data)
                        for i in xrange(nsamples):
                            reps.append(sess.run(internal_rep,feed_dict={input_ph: x_data[i].reshape([4,1])}).flatten())
                        item_rep_similarity = np.zeros([nsamples,nsamples])
                        for i in xrange(nsamples):
                            for j in xrange(i,nsamples):
                                item_rep_similarity[i,j] = np.linalg.norm(reps[i]-reps[j]) 
                                item_rep_similarity[j,i] = item_rep_similarity[i,j]
                        plt.imshow(item_rep_similarity,cmap='Greys_r',interpolation='none') #cosine distance
                        plt.show()


                    def train_with_standard_loss():
                        sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: x_data,target_ph: y_data})

                    def train_with_linearized_loss():
                        sess.run(linearized_train,feed_dict={eta_ph: curr_eta,input_ph: x_data,target_ph: y_data})

                    print "Initial MSE: %f" %(test_accuracy())

                    #loaded_pre_outputs = np.loadtxt(pre_output_filename_to_load,delimiter=',')

                    curr_eta = init_eta
                    rep_track = []
                    loss_filename = filename_prefix + "loss_track.csv"
                    #if os.path.exists(filename):
                #	os.remove(filename)
#                    save_activations(pre_output,filename_prefix+"initial_pre_outputs.csv")
#                    save_activations(internal_rep,filename_prefix+"initial_reps.csv")
                    with open(loss_filename, 'w') as fout:
                        fout.write("epoch, MSE\n")
                        curr_mse = test_accuracy()
                        fout.write("%i, %f\n" %(0, curr_mse))
                        for epoch in xrange(nepochs):
                            if network == 'nonlinear':
                                train_with_standard_loss()
                            else:
                                train_with_linearized_loss()
                            if epoch % 5 == 0:
                                curr_mse = test_accuracy()
                                print "epoch: %i, MSE: %f" %(epoch, curr_mse)	
                                fout.write("%i, %f\n" %(epoch, curr_mse))
#                            if epoch % 100 == 0:
#                                save_activations(internal_rep,filename_prefix+"epoch_%i_internal_rep.csv" %epoch)
#                                save_activations(pre_output,filename_prefix+"epoch_%i_pre_outputs.csv" %epoch)
                            
                            if epoch % eta_decay_epoch == 0:
                                curr_eta *= eta_decay
                        

                    print "Final MSE: %f" %(test_accuracy())

#                    print_preoutputs()
#                    save_activations(pre_output,filename_prefix+"final_pre_outputs.csv")
#                    save_activations(internal_rep,filename_prefix+"final_reps.csv")
