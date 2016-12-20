import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import os

######Parameters###################
init_eta = 0.05
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 500
nhidden = 2

rseed = 0 #reproducibility
filename = "standard_rep_tracks_rseed_%i.csv" %rseed
###################################

x_data = numpy.array([numpy.roll([1,0,0,0],i) for i in xrange(4)])
y_data = numpy.array(map(lambda x: [1*(x[0] or x[1]),x[0],x[1],1*(x[2] or x[3]),x[2],x[3]],x_data))
y_data = y_data.reshape([len(x_data),6])
print "x_data:"
print x_data
print

print "y_data:"
print x_data
print

numpy.random.seed(rseed)
tf.set_random_seed(rseed)

input_ph = tf.placeholder(tf.float32, shape=[4,1])
target_ph =  tf.placeholder(tf.float32, shape=[6,1])
W1 = tf.Variable(tf.random_normal([nhidden,4],0,0.1))
b1 = tf.Variable(tf.random_normal([nhidden,1],1,0.1))
W2 = tf.Variable(tf.random_normal([6,nhidden],0,0.1))
b2 = tf.Variable(tf.random_normal([6,1],1,0.1))
internal_rep = tf.matmul(W1,input_ph)+b1
output = tf.nn.relu(tf.matmul(W2,internal_rep)+b2)
pre_output = (tf.matmul(W2,internal_rep)+b2)

rep_mean_ph =  tf.placeholder(tf.float32, shape=[nhidden,1])

loss = tf.reduce_sum(tf.square(output - target_ph))# +0.05*(tf.nn.l2_loss(internal_rep))
linearized_loss = tf.reduce_sum(tf.square(pre_output - target_ph))# +0.05*(tf.nn.l2_loss(internal_rep))
eta_ph = tf.placeholder(tf.float32)
optimizer = tf.train.GradientDescentOptimizer(eta_ph)
train = optimizer.minimize(loss)
linearized_train = optimizer.minimize(linearized_loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

def test_accuracy():
    MSE = 0.0
    for i in xrange(len(x_data)):
	MSE += sess.run(loss,feed_dict={input_ph: x_data[i].reshape([4,1]),target_ph: y_data[i].reshape([6,1])})
    MSE /= 4.0
    return MSE

def print_outputs():
    for i in xrange(len(x_data)):
	print x_data[i], y_data[i], sess.run(output,feed_dict={input_ph: x_data[i].reshape([4,1])}).flatten()

def print_preoutputs():
    for i in xrange(len(x_data)):
	print x_data[i], y_data[i], sess.run(pre_output,feed_dict={input_ph: x_data[i].reshape([4,1])}).flatten()

def print_reps():
    for i in xrange(len(x_data)):
	print x_data[i], y_data[i], sess.run(internal_rep,feed_dict={input_ph: x_data[i].reshape([4,1])}).flatten()

def get_reps():
    reps = []
    nsamples = len(x_data)
    for i in xrange(nsamples):
	reps.append(sess.run(internal_rep,feed_dict={input_ph: x_data[i].reshape([4,1])}).flatten())
    return reps

def save_activations(tf_object,filename,remove_old=True):
    if remove_old and os.path.exists(filename):
	print "removing..."
	os.remove(filename)
    with open(filename,'ab') as fout:
	for i in xrange(len(x_data)):
	    numpy.savetxt(fout,sess.run(tf_object,feed_dict={input_ph: x_data[i].reshape([4,1])}).reshape((1,6)),delimiter=',')


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
    training_order = numpy.random.permutation(len(x_data))
    for example_i in training_order:
	sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: x_data[example_i].reshape([4,1]),target_ph: y_data[example_i].reshape([6,1])})

print "Initial MSE: %f" %(test_accuracy())

curr_eta = init_eta
rep_track = []
fout = open(filename,'ab')
for epoch in xrange(nepochs):
    train_with_standard_loss()
#    train_with_difference_penalty()
    if epoch % 10 == 0:
	numpy.savetxt(fout,numpy.array(get_reps()).flatten(),delimiter=',')
    if epoch % 10 == 0:
	print "epoch: %i, MSE: %f" %(epoch, test_accuracy())	
    if epoch % 100 == 0:
	print_reps()	
#	display_rep_similarity()
    if epoch % eta_decay_epoch == 0:
	curr_eta *= eta_decay
fout.close()
    

print "Final MSE: %f" %(test_accuracy())

print_preoutputs()
save_activations(pre_output,"final_preoutputs_rseed_%i.csv" %rseed)
