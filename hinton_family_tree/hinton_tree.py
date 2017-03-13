import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import os

######Parameters###################
init_eta = 0.005
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 1000
nhidden_separate = 12
nhidden_shared = 12
npeople_per = 12
nrelationships_per = 12
nfamilies = 2

#rseed = 2  #reproducibility
###################################

#build tree
gender = {"Christopher":"M",
    "Penelope":"F",
    "Margaret":"F",
    "Arthur":"M",
    "Victoria":"F",
    "James":"M",
    "Colin":"M",
    "Charlotte":"F",
    "Andrew":"M",
    "Christine":"F",
    "Jennifer":"F",
    "Charles":"M"
}


#H = husband, W = wife, F = father, M = mother, Son = son, D = daughter, U = uncle, A = aunt, B = brother, Sis = sister, Nep = nephew, Nie = niece
relationship_dict = {
("Christopher","Penelope"): "H",
("Arthur","Margaret"): "H",
("James","Victoria"): "H",
("Andrew","Christine"): "H",
("Charles","Jennifer"): "H",
("Christopher","Arthur"): "F",
("Christopher","Victoria"): "F",
("James","Colin"): "F", 
("James","Charlotte"): "F", 
("Andrew","James"): "F",
("Andrew","Jennifer"): "F"
}

reverse_pair = lambda x: (x[1],x[0])

def fill_out_relationships(relationship_dict):
    people = gender.keys()
    #Wives, mothers sons, daughters
    for key in relationship_dict.keys(): 
	if relationship_dict[key] == "H":
	    relationship_dict[reverse_pair(key)] = "W"	
	elif relationship_dict[key] == "F": 
	    wife = None
	    for person in people:
		 if (key[0],person) in relationship_dict.keys() and relationship_dict[(key[0],person)] == "H":
		    wife = person
		    break
	    mother_key = (wife,key[1]) 
	    relationship_dict[mother_key] = "M"
	    if gender[key[1]] == "M":
		relationship_dict[reverse_pair(key)] = "Son"
		relationship_dict[reverse_pair(mother_key)] = "Son"
	    else: 
		relationship_dict[reverse_pair(key)] = "D"
		relationship_dict[reverse_pair(mother_key)] = "D"
    #Brothers, sisters
    for person in people:
	father = None
	for person_2 in people:	
	     if (person_2,person) in relationship_dict.keys() and relationship_dict[(person_2,person)] == "F":
		father = person_2
		break
	for person_2 in people:
	    if person_2 == person:
		continue
	    if (father,person_2) in relationship_dict.keys() and relationship_dict[(father,person_2)] == "F": #same father
		if gender[person] == "M":
		    relationship_dict[(person,person_2)] = "B"	
		    relationship_dict[(person_2,person)] = "Sis"	
		else:
		    relationship_dict[(person,person_2)] = "Sis"	
		    relationship_dict[(person_2,person)] = "B"	
		break
    #Uncles, Aunts
    for person in people:
	father = None
	mother = None
	for person_2 in people:	
	     if (person_2,person) in relationship_dict.keys():
		if relationship_dict[(person_2,person)] == "F":
		    father = person_2
		elif relationship_dict[(person_2,person)] == "M":
		    mother = person_2
	aunts_and_uncles = []
	for person_2 in people:
	    if person_2 in [person,father,mother]:
		continue
	    if ((father,person_2) in relationship_dict.keys() and relationship_dict[(person_2,father)] == "Sis") or ((mother,person_2) in relationship_dict.keys() and relationship_dict[(person_2,mother)] == "Sis"): #aunt
		aunts_and_uncles.append(person_2)
		relationship_dict[(person_2,person)] = "A"	
		if gender[person] == "M":
		    relationship_dict[(person,person_2)] = "Nep"	
		else:
		    relationship_dict[(person,person_2)] = "Nie"	
	    elif ((father,person_2) in relationship_dict.keys() and relationship_dict[(person_2,father)] == "B") or ((mother,person_2) in relationship_dict.keys() and relationship_dict[(person_2,mother)] == "B"): #uncle
		aunts_and_uncles.append(person_2)
		relationship_dict[(person_2,person)] = "U"	
		if gender[person] == "M":
		    relationship_dict[(person,person_2)] = "Nep"	
		else:
		    relationship_dict[(person,person_2)] = "Nie"	
	#Now aunts and uncles by marriage	
	for person_2 in people:
	    if person_2 in [person,father,mother]:
		continue
	    for person_3 in aunts_and_uncles:
		if (person_2,person_3) in relationship_dict.keys() and relationship_dict[(person_2,person_3)] in ["H","W"]:
		    if gender[person_2] == "M":
			relationship_dict[(person_2,person)] = "U"	
			if gender[person] == "M":
			    relationship_dict[(person,person_2)] = "Nep"	
			else:
			    relationship_dict[(person,person_2)] = "Nie"	
		    else:
			relationship_dict[(person_2,person)] = "A"	
			if gender[person] == "M":
			    relationship_dict[(person,person_2)] = "Nep"	
			else:
			    relationship_dict[(person,person_2)] = "Nie"	
    return relationship_dict

relationship_dict = fill_out_relationships(relationship_dict)

def restructure_dict(x):
    """moves from (person1,person2) -> relationship dict to (person2,relationship) -> [persona,personb,...] dict"""
    new_dict = {}
    for key in x.keys():
	new_key = (key[1],x[key])
	if new_key in new_dict.keys():
	    new_dict[new_key].append(key[0])	
	else:
	    new_dict[new_key] = [key[0]]	
    return new_dict

restructured_relationship_dict = restructure_dict(relationship_dict)
print restructured_relationship_dict.keys()
print restructured_relationship_dict
print 
print

#Now build single family's data matrix
restructured_relationship_dict_keys = restructured_relationship_dict.keys()
num_relations = len(restructured_relationship_dict_keys)
one_family_input_matrix = numpy.zeros((num_relations,npeople_per+nrelationships_per)) 
one_family_output_matrix = numpy.zeros((num_relations,npeople_per)) 
people = gender.keys()
relationships = ["F","M","H","W","Son","D","U","A","B","Sis","Nep","Nie"]

print people
print relationships
for i in xrange(len(restructured_relationship_dict_keys)):
    key = restructured_relationship_dict_keys[i]
    one_family_input_matrix[i,people.index(key[0])] = 1
    one_family_input_matrix[i,npeople_per+relationships.index(key[1])] = 1
    for person in restructured_relationship_dict[key]:
	one_family_output_matrix[i,people.index(person)] = 1
    
    
#print one_family_input_matrix
#print one_family_output_matrix 
input_matrix_shape = [nfamilies*i for i in one_family_input_matrix.shape]
output_matrix_shape = [nfamilies*i for i in one_family_output_matrix.shape]
input_shape = input_matrix_shape[1]
output_shape = output_matrix_shape[1]
x_data = numpy.zeros((0,input_shape))
y_data = numpy.zeros((0,output_shape))


for i in xrange(nfamilies):
    this_x = numpy.concatenate([numpy.zeros((num_relations,i*(npeople_per+nrelationships_per))), one_family_input_matrix ,numpy.zeros((num_relations,(nfamilies-(i+1))*(npeople_per+nrelationships_per)))],1)
    x_data = numpy.concatenate([x_data,this_x],0)
    this_y = numpy.concatenate([numpy.zeros((num_relations,i*(npeople_per))), one_family_output_matrix ,numpy.zeros((num_relations,(nfamilies-(i+1))*(npeople_per)))],1)
    y_data = numpy.concatenate([y_data,this_y],0)


#numpy.savetxt("one_family_input.csv",one_family_input_matrix,delimiter=',')
#numpy.savetxt("one_family_output.csv",one_family_output_matrix,delimiter=',')
#numpy.savetxt("hinton_x_data.csv",x_data,delimiter=',')
#numpy.savetxt("hinton_y_data.csv",y_data,delimiter=',')


print "x_data shape:"
print x_data.shape
print

print "y_data shape:"
print y_data.shape
print


for rseed in xrange(100):
    print "run %i" %rseed
    filename_prefix = "results/hinton_nonlinear_smallweights_nhidden_%i_rseed_%i_" %(nhidden_shared,rseed)

    numpy.random.seed(rseed)
    tf.set_random_seed(rseed)

    input_ph = tf.placeholder(tf.float32, shape=[input_shape,None])
    #handle some reshaping to get from block diagonal data matrix structure to desired input structure
    split_inputs = tf.split(input_ph,[(npeople_per if (i % 2 == 0) else nrelationships_per) for i in xrange(nfamilies*2)],0)
    people_inputs = [this_input for (i,this_input) in enumerate(split_inputs) if i % 2 == 0]
    relationship_inputs = [this_input for (i,this_input) in enumerate(split_inputs) if i % 2 == 1]
    people_input = tf.concat(people_inputs,axis=0)
    relationship_input = tf.concat(relationship_inputs,axis=0)
    #build the network
     ############Working simpler network with eta = 0.005, nhidden = 13 
    W1 = tf.Variable(tf.random_uniform([nhidden_shared,input_shape],0,0.001))
    b1 = tf.Variable(tf.random_uniform([nhidden_shared,1],0,0.001))
    W2 = tf.Variable(tf.random_uniform([output_shape,nhidden_shared],0,0.001))
    b2 = tf.Variable(tf.random_uniform([output_shape,1],0,0.001))
    pre_middle_rep = tf.matmul(W1,tf.concat([people_input,relationship_input],0))+b1
    middle_rep = tf.nn.relu(pre_middle_rep)

    pre_output = tf.matmul(W2,middle_rep)+b2

    output = tf.nn.relu(pre_output)


    target_ph =  tf.placeholder(tf.float32, shape=[output_shape,None])
    pre_output_target_ph =  tf.placeholder(tf.float32, shape=[output_shape,None])

    loss = tf.reduce_sum(tf.square(output - target_ph))
    linearized_loss = tf.reduce_sum(tf.square(pre_output - pre_output_target_ph))
    eta_ph = tf.placeholder(tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(eta_ph)
    train = optimizer.minimize(loss)
    linearized_train = optimizer.minimize(linearized_loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    def test_accuracy():
	MSE = 0.0
	for i in xrange(len(x_data)):
	    MSE += sess.run(loss,feed_dict={input_ph: x_data[i].reshape([input_shape,1]),target_ph: y_data[i].reshape([output_shape,1])})
	MSE /= len(x_data)
	return MSE

    def print_outputs():
	for i in xrange(len(x_data)):
	    print x_data[i], y_data[i], sess.run(output,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten()

    def print_preoutputs():
	for i in xrange(len(x_data)):
	    print x_data[i], y_data[i], sess.run(pre_output,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten()

    def print_reps():
	for i in xrange(len(x_data)):
	    print x_data[i], y_data[i], sess.run(middle_rep,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten()

    def get_reps():
	reps = []
	nsamples = len(x_data)
	for i in xrange(nsamples):
	    reps.append(sess.run(middle_rep,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten())
	return reps

    def save_activations(tf_object,filename,remove_old=True):
	if remove_old and os.path.exists(filename):
	    os.remove(filename)
	with open(filename,'ab') as fout:
	    for i in xrange(len(x_data)):
		numpy.savetxt(fout,sess.run(tf_object,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).reshape((1,-1)),delimiter=',')

    def save_weights(tf_object,filename,remove_old=True):
	if remove_old and os.path.exists(filename):
	    os.remove(filename)
	with open(filename,'ab') as fout:
	    numpy.savetxt(fout,sess.run(tf_object),delimiter=',')

    def display_rep_similarity():
	reps = []
	nsamples = len(x_data)
	for i in xrange(nsamples):
	    reps.append(sess.run(pre_middle_rep,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten())
	item_rep_similarity = numpy.zeros([nsamples,nsamples])
	for i in xrange(nsamples):
	    for j in xrange(i,nsamples):
		item_rep_similarity[i,j] = numpy.linalg.norm(reps[i]-reps[j]) 
		item_rep_similarity[j,i] = item_rep_similarity[i,j]
	plt.imshow(item_rep_similarity,cmap='Greys_r',interpolation='none') #cosine distance
	plt.show()

    def display_po_similarity():
	reps = []
	nsamples = len(x_data)
	for i in xrange(nsamples):
	    reps.append(sess.run(pre_output,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten())
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
	    sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: x_data[example_i].reshape([input_shape,1]),target_ph: y_data[example_i].reshape([output_shape,1])})

    def batch_train_with_standard_loss():
	sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: x_data.transpose(),target_ph: y_data.transpose()})

    def train_with_linearized_loss(targets):
	training_order = numpy.random.permutation(len(x_data))
	for example_i in training_order:
	    sess.run(linearized_train,feed_dict={eta_ph: curr_eta,input_ph: x_data[example_i].reshape([input_shape,1]),pre_output_target_ph: targets[example_i].reshape([output_shape,1])})

    print "Initial MSE: %f" %(test_accuracy())

    #loaded_pre_outputs = numpy.loadtxt(pre_output_filename_to_load,delimiter=',')

    curr_eta = init_eta
    rep_track = []
    filename = filename_prefix + "rep_tracks.csv"
    if os.path.exists(filename):
	os.remove(filename)
    fout = open(filename,'ab')
    for epoch in xrange(nepochs):
#        batch_train_with_standard_loss()
        train_with_standard_loss()
    #    train_with_linearized_loss(loaded_pre_outputs)
#	train_with_linearized_loss(y_data)
#	if epoch % 10 == 0:
#	    save_activations(pre_output,filename_prefix+"epoch_%i_pre_outputs.csv" %epoch)
#	    save_activations(middle_rep,filename_prefix+"epoch_%i_middle_rep.csv" %epoch)
#	    save_weights(W1,filename_prefix+"epoch_%i_W1.csv" %epoch)
#	    save_weights(W2,filename_prefix+"epoch_%i_W2.csv" %epoch)
    #	numpy.savetxt(fout,numpy.array(get_reps()),delimiter=',')
	if epoch % 10 == 0:
	    temp = test_accuracy()
	    print "epoch: %i, MSE: %f" %(epoch, temp)	
	    numpy.savetxt(fout,[temp],delimiter=',')
#	if epoch % 100 == 0:
#	    display_rep_similarity()
#	    display_po_similarity()
	if epoch % eta_decay_epoch == 0:
	    curr_eta *= eta_decay
    fout.close()
	

    print "Final MSE: %f" %(test_accuracy())

#    print_preoutputs()
#    display_rep_similarity()
#    display_po_similarity()
#    save_activations(pre_middle_rep,filename_prefix+"pre_middle_reps.csv")
#    save_activations(pre_output,filename_prefix+"pre_outputs.csv")
