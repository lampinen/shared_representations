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
input_matrix_shape = one_family_input_matrix.shape
output_matrix_shape = one_family_output_matrix.shape
input_shape = input_matrix_shape[1]
output_shape = output_matrix_shape[1]
x_data = one_family_input_matrix
y_data = one_family_output_matrix 

bin_data = [(bin(x)[2:]) for x in xrange(12)] 
bin_data = numpy.array(map(lambda x: [0]*(12-len(x))+(map(int,x)),bin_data))
alt_y_data = map(lambda x: bin_data[numpy.argmax(x)], x_data)



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
    filename_prefix = "results/sequential_noshared/SN_hinton_nhidden_%i_rseed_%i_" %(nhidden_shared,rseed)

    numpy.random.seed(rseed)
    tf.set_random_seed(rseed)

    input_ph = tf.placeholder(tf.float32, shape=[input_shape,None])

    ############build network#################### 
    W1f1 = tf.Variable(tf.random_uniform([nhidden_shared,input_shape],0,0.1))
    W1f2 = tf.Variable(tf.random_uniform([nhidden_shared,input_shape],0,0.1))
    b1 = tf.Variable(tf.random_uniform([nhidden_shared,1],0,0.1))
    W2 = tf.Variable(tf.random_uniform([nhidden_shared,nhidden_shared],0,0.1)) 
    b2 = tf.Variable(tf.random_uniform([nhidden_shared,1],0,0.1))
    W3f1 = tf.Variable(tf.random_uniform([output_shape,nhidden_shared],0,0.1))
    W3f2 = tf.Variable(tf.random_uniform([output_shape,nhidden_shared],0,0.1))
    b3f1 = tf.Variable(tf.random_uniform([output_shape,1],0,0.1))
    b3f2 = tf.Variable(tf.random_uniform([output_shape,1],0,0.1))


    f1_pre_middle_rep = tf.matmul(W1f1,input_ph)+b1
    f2_pre_middle_rep = tf.matmul(W1f2,input_ph)+b1
    f1_middle_rep = tf.nn.relu(f1_pre_middle_rep)
    f2_middle_rep = tf.nn.relu(f2_pre_middle_rep)

#    f1_pre_output = tf.matmul(W3f1,tf.nn.relu(tf.matmul(W2,f1_middle_rep)+b2))+b3f1
#    f2_pre_output = tf.matmul(W3f2,tf.nn.relu(tf.matmul(W2,f2_middle_rep)+b2))+b3f2

    f1_pre_output = tf.matmul(W3f1,f1_middle_rep)+b3f1
    f2_pre_output = tf.matmul(W3f2,f2_middle_rep)+b3f2

    f1_output = tf.nn.relu(f1_pre_output)
    f2_output = tf.nn.relu(f2_pre_output)

    target_ph =  tf.placeholder(tf.float32, shape=[output_shape,None])

    f1_loss = tf.reduce_sum(tf.square(f1_output - target_ph))
    f2_loss = tf.reduce_sum(tf.square(f2_output - target_ph))
#    linearized_loss = tf.reduce_sum(tf.square(pre_output - pre_output_target_ph))
    eta_ph = tf.placeholder(tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(eta_ph)
    f1_train = optimizer.minimize(f1_loss)
    f2_train = optimizer.minimize(f2_loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    def test_domain_accuracy(this_domain):
	MSE = 0.0
	order = numpy.array(range(len(x_data)))
	for i in order:
	    if this_domain == 1:
		MSE += sess.run(f1_loss,feed_dict={input_ph: x_data[i].reshape([input_shape,1]),target_ph: y_data[i].reshape([output_shape,1])})
	    else:
		MSE += sess.run(f2_loss,feed_dict={input_ph: x_data[i].reshape([input_shape,1]),target_ph: y_data[i].reshape([output_shape,1])})
	MSE /= len(x_data)
	return MSE

    def print_outputs(domain):
	for i in xrange(1,2):
	    if domain == 1:
		print x_data[i], y_data[i], sess.run(f1_output,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten()
	    else:
		print x_data[i], y_data[i], sess.run(f2_output,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten()

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

#    def display_rep_similarity():
#	reps = []
#	nsamples = len(x_data)
#	for i in xrange(nsamples):
#	    reps.append(sess.run(pre_middle_rep,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten())
#	item_rep_similarity = numpy.zeros([nsamples,nsamples])
#	for i in xrange(nsamples):
#	    for j in xrange(i,nsamples):
#		item_rep_similarity[i,j] = numpy.linalg.norm(reps[i]-reps[j]) 
#		item_rep_similarity[j,i] = item_rep_similarity[i,j]
#	plt.imshow(item_rep_similarity,cmap='Greys_r',interpolation='none') #cosine distance
#	plt.show()
#
#    def display_po_similarity():
#	reps = []
#	nsamples = len(x_data)
#	for i in xrange(nsamples):
#	    reps.append(sess.run(pre_output,feed_dict={input_ph: x_data[i].reshape([input_shape,1])}).flatten())
#	item_rep_similarity = numpy.zeros([nsamples,nsamples])
#	for i in xrange(nsamples):
#	    for j in xrange(i,nsamples):
#		item_rep_similarity[i,j] = numpy.linalg.norm(reps[i]-reps[j]) 
#		item_rep_similarity[j,i] = item_rep_similarity[i,j]
#	plt.imshow(item_rep_similarity,cmap='Greys_r',interpolation='none') #cosine distance
#	plt.show()

    def train_domain_with_standard_loss(this_domain):
	training_order = numpy.random.permutation(len(x_data))
	for example_i in training_order:
	    if this_domain == 1:
		sess.run(f1_train,feed_dict={eta_ph: curr_eta,input_ph: x_data[example_i].reshape([input_shape,1]),target_ph: y_data[example_i].reshape([output_shape,1])})
	    else:
		sess.run(f2_train,feed_dict={eta_ph: curr_eta,input_ph: x_data[example_i].reshape([input_shape,1]),target_ph: y_data[example_i].reshape([output_shape,1])})


    print "Initial MSEa: %f, %f" %(test_domain_accuracy(1),test_domain_accuracy(2))

    #loaded_pre_outputs = numpy.loadtxt(pre_output_filename_to_load,delimiter=',')

    curr_eta = init_eta
    rep_track = []
    filename = filename_prefix + "rep_tracks.csv"
    if os.path.exists(filename):
	os.remove(filename)
    fout = open(filename,'ab')
    saved = False
    for epoch in xrange(nepochs):
#	train_domain_with_standard_loss(1)
#	train_domain_with_standard_loss(2)
#	if epoch % 10 == 0:
#	    curr_error = test_domain_accuracy(2)
#	    print "epoch: %i, family 2 MSE: %f" %(epoch, curr_error)	
#	    if (not saved) and curr_error <= 0.05:
#		save_activations(f1_pre_middle_rep,filename_prefix+"f1_pre_middle_reps.csv")
#		save_activations(f1_pre_output,filename_prefix+"f1_pre_outputs.csv")
#		save_activations(f2_pre_middle_rep,filename_prefix+"f2_pre_middle_reps.csv")
#		save_activations(f2_pre_output,filename_prefix+"f2_pre_outputs.csv")
	if epoch < nepochs/2:  
	    train_domain_with_standard_loss(1)
	    if epoch % 10 == 0:
		curr_error = test_domain_accuracy(1)
		print "epoch: %i, family 1 MSE: %f" %(epoch, curr_error)	
	else:
	    train_domain_with_standard_loss(2)
	    if epoch % 10 == 0:
		curr_error = test_domain_accuracy(2)
		print "epoch: %i, family 2 MSE: %f" %(epoch, curr_error)		    
	fout.write(str(curr_error)+',')

#	if epoch % 100 == 0:
#	    display_rep_similarity()
#	    display_po_similarity()
	if epoch % eta_decay_epoch == 0:
	    curr_eta *= eta_decay
	if epoch == nepochs/2:
	    curr_eta = init_eta
    fout.close()
	

    print "final MSEs: %f, %f" %(test_domain_accuracy(1),test_domain_accuracy(2))

#    print_preoutputs()
#    display_rep_similarity()
#    display_po_similarity()
    save_activations(f1_pre_output,filename_prefix +'f1_pre_outputs.csv',remove_old=True)
    save_activations(f2_pre_output,filename_prefix +'f2_pre_outputs.csv',remove_old=True)
    save_activations(f1_pre_middle_rep,filename_prefix +'f1_pre_middle_reps.csv',remove_old=True)
    save_activations(f2_pre_middle_rep,filename_prefix +'f2_pre_middle_reps.csv',remove_old=True)
