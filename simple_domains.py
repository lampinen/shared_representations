import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import os

######Parameters###################
init_eta = 0.01
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 10000
nruns = 5
#nhidden = 6
#rseed = 2  #reproducibility
###################################



x_data = {}
y_data = {}
x_data[1] = numpy.array([[1,0],[0,1],[0,0],[0,0] ])
this_y_data = numpy.array(map(lambda x: [1*(x[0] or x[1]),x[0],x[1]],x_data[1]))
y_data[1] = this_y_data.reshape([len(x_data[1]),3])
x_data[2] = numpy.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] ])
this_y_data = numpy.array(map(lambda x: [1*(x[0] or x[1]),x[0],x[1],1*(x[2] or x[3]),x[2],x[3]],x_data[2]))
y_data[2] = this_y_data.reshape([len(x_data[2]),6])
x_data[3] = numpy.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
this_y_data = numpy.array(map(lambda x: [1*(x[0] or x[1]),x[0],x[1],1*(x[2] or x[3]),x[2],x[3],1*(x[4] or x[5]),x[4],x[5]],x_data[3]))
y_data[3] = this_y_data.reshape([len(x_data[3]),9])
print "x_data:"
print x_data
print

print "y_data:"
print y_data
print



for network in ['nonlinear', 'linear']:
    for nlayer in [3,2]:
	for ndomains in [1,2,3]:
	    ninput = 2*ndomains
	    noutput = 3*ndomains 
	    nhidden = ninput
	    for rseed in xrange(2, nruns):
		print "nlayer %i ndomains %i run %i" % (nlayer, ndomains, rseed)
		filename_prefix = "results/depth_and_ndom_comp/%s_nlayer_%i_ndomains_%i_rseed_%i_" %(network,nlayer,ndomains,rseed)

		numpy.random.seed(rseed)
		tf.set_random_seed(rseed)

		input_ph = tf.placeholder(tf.float32, shape=[2*ndomains,None])
		target_ph = tf.placeholder(tf.float32, shape=[noutput,None])
		if nlayer == 2:
		    W1 = tf.Variable(tf.random_uniform([nhidden,ninput],0.,2.0/(nhidden+ninput)))
		    W2 = tf.Variable(tf.random_uniform([noutput,nhidden],0.,2./(nhidden+noutput)))
		    internal_rep = tf.matmul(W1,input_ph)
		    pre_output = tf.matmul(W2,internal_rep)
		elif nlayer == 3:
		    W1 = tf.Variable(tf.random_uniform([nhidden,ninput],0.,2./(nhidden+ninput)))
		    W2 = tf.Variable(tf.random_uniform([nhidden,nhidden],0.,1./nhidden))
		    W3 = tf.Variable(tf.random_uniform([noutput,nhidden],0.,2./(nhidden+noutput)))
		    internal_rep = tf.matmul(W1,input_ph)
			
		    pre_output = tf.matmul(W3,tf.matmul(W2,internal_rep))
		elif nlayer == 4:
		    W1 = tf.Variable(tf.random_uniform([nhidden,ninput],0.,2./(nhidden+ninput)))
		    W2 = tf.Variable(tf.random_uniform([nhidden,nhidden],0.,1./nhidden))
		    W3 = tf.Variable(tf.random_uniform([nhidden,nhidden],0.,1./nhidden))
		    W4 = tf.Variable(tf.random_uniform([noutput,nhidden],0.,2./(nhidden+noutput)))
		    internal_rep = tf.matmul(W1,input_ph)
			
		    pre_output = tf.matmul(W4,tf.matmul(W3,tf.matmul(W2,internal_rep)))
		else:
		    print "Error, invalid number of layers given"
		    exit(1)

		output = tf.nn.relu(pre_output)
		rep_mean_ph =  tf.placeholder(tf.float32, shape=[nhidden,1])

		loss = tf.reduce_sum(tf.square(output - target_ph))# +0.05*(tf.nn.l2_loss(internal_rep))
		linearized_loss = tf.reduce_sum(tf.square(pre_output - target_ph))# +0.05*(tf.nn.l2_loss(internal_rep))
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
		    MSE = 0.0
		    for i in xrange(len(x_data[ndomains])):
			MSE += sess.run(loss,feed_dict={input_ph: x_data[ndomains][i].reshape([ninput,1]),target_ph: y_data[ndomains][i].reshape([noutput,1])})
		    MSE /= len(x_data)
		    return MSE

		def print_outputs():
		    for i in xrange(len(x_data[ndomains])):
			print x_data[ndomains][i], y_data[ndomains][i], sess.run(output,feed_dict={input_ph: x_data[ndomains][i].reshape([ninput,1])}).flatten()

		def print_preoutputs():
		    for i in xrange(len(x_data[ndomains])):
			print x_data[ndomains][i], y_data[ndomains][i], sess.run(pre_output,feed_dict={input_ph: x_data[ndomains][i].reshape([ninput,1])}).flatten()


		def get_reps():
		    reps = []
		    nsamples = len(x_data)
		    for i in xrange(nsamples):
			reps.append(sess.run(internal_rep,feed_dict={input_ph: x_data[ndomains][i].reshape([ninput,1])}).flatten())
		    return reps

		def save_activations(tf_object,filename,remove_old=True):
		    if remove_old and os.path.exists(filename):
			os.remove(filename)
		    with open(filename,'ab') as fout:
			for i in xrange(len(x_data[ndomains])):
			    numpy.savetxt(fout,sess.run(tf_object,feed_dict={input_ph: x_data[ndomains][i].reshape([ninput,1])}).reshape((1,-1)),delimiter=',')

		def save_weights(tf_object,filename,remove_old=True):
		    if remove_old and os.path.exists(filename):
			os.remove(filename)
		    with open(filename,'ab') as fout:
			numpy.savetxt(fout,sess.run(tf_object),delimiter=',')

		def display_rep_similarity():
		    reps = []
		    nsamples = len(x_data)
		    for i in xrange(nsamples):
			reps.append(sess.run(internal_rep,feed_dict={input_ph: x_data[i].reshape([4,1])}).flatten())
		    item_rep_similarity = numpy.zeros([nsamples,nsamples])
		    for i in xrange(nsamples):
			for j in xrange(i,nsamples):
			    item_rep_similarity[i,j] = numpy.linalg.norm(reps[i]-reps[j]) 
			    item_rep_similarity[j,i] = item_rep_similarity[i,j]
		    plt.imshow(item_rep_similarity,cmap='Greys_r',interpolation='none') #cosine distance
		    plt.show()


		def train_with_standard_loss():
		    training_order = numpy.random.permutation(len(x_data[ndomains]))
		    for example_i in training_order:
			sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: x_data[ndomains][example_i].reshape([ninput,1]),target_ph: y_data[ndomains][example_i].reshape([noutput,1])})

		def train_with_linearized_loss():
		    training_order = numpy.random.permutation(len(x_data[ndomains]))
		    for example_i in training_order:
			sess.run(linearized_train,feed_dict={eta_ph: curr_eta,input_ph: x_data[ndomains][example_i].reshape([ninput,1]),target_ph: y_data[ndomains][example_i].reshape([noutput,1])})

		print "Initial MSE: %f" %(test_accuracy())

		#loaded_pre_outputs = numpy.loadtxt(pre_output_filename_to_load,delimiter=',')

		curr_eta = init_eta
		rep_track = []
		loss_filename = filename_prefix + "loss_track.csv"
		#if os.path.exists(filename):
	    #	os.remove(filename)
		save_activations(pre_output,filename_prefix+"initial_pre_outputs.csv")
		save_activations(internal_rep,filename_prefix+"initial_reps.csv")
		with open(loss_filename, 'w') as fout:
		    fout.write("epoch, MSE\n")
		    for epoch in xrange(nepochs):
			if network == 'nonlinear':
			    train_with_standard_loss()
			else:
			    train_with_linearized_loss()
			if epoch % 5 == 0:
			    curr_mse = test_accuracy()
			    print "epoch: %i, MSE: %f" %(epoch, curr_mse)	
			    fout.write("%i, %f\n" %(epoch, curr_mse))
			if epoch % 100 == 0:
			    save_activations(internal_rep,filename_prefix+"epoch_%i_internal_rep.csv" %epoch)
			    save_activations(pre_output,filename_prefix+"epoch_%i_pre_outputs.csv" %epoch)
			
			if epoch % eta_decay_epoch == 0:
			    curr_eta *= eta_decay
		    

		print "Final MSE: %f" %(test_accuracy())

		print_preoutputs()
		save_activations(pre_output,filename_prefix+"final_pre_outputs.csv")
		save_activations(internal_rep,filename_prefix+"final_reps.csv")
