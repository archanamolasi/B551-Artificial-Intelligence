# Author: Archana, Prerna, Shruti
# Updated date: Dec 3, 2016

# This programs finds the correct orientation for an image file.
# (1) We have implemented three methods of classifying the image orientation.
# a) Nearest neighbor: This method calculates the euclidean distance between a test image vector and all the training image vectors; and
# assigns the orientation of closest training image to the test image.
# b) Adaboost: This method uses randomly generated tuple of two indices of the training image as decision stumps.
# c) Neural  network: This method implements a fully connected feed forward neural network with one hidden layer.
# (2) We create two dictionaries of training and testing data.
# We have implemented the three methods to classify the image orientation.
# In the end confusion matrix and accuracy is calculated. We have also created an output file with image id and its derived orientation.
# Time limits: We have experimented with only following time limits for each method.
# Nearest = 26 mins
# Adaboost = 10 mins
# Neural Net = 20 mins
# (3) Assumptions and problems:
# For the Neural network the sigmoid activation function was always returning 1 for our input. So we normalised the input to the 
# activation function such that the sigmoid returns a reasonable value within its domain.


#import modules
import collections,sys,math, json,copy
import random as rand
from scipy.spatial import distance
import numpy as np
from numpy import array
import operator

# global variables definition
infinity=999999999999999999
conf_mat = {"0": [0, 0, 0, 0], "90":  [0, 0, 0, 0],"180":  [0, 0, 0, 0],"270":  [0, 0, 0, 0]} 

# This function claculates the accuracy from the confusion matrix
def calc_accuracy(conf_mat):
	acc_num=conf_mat["0"][0]+conf_mat["90"][1]+conf_mat["180"][2]+conf_mat["270"][3]
	print "Accuracy is :", 100*acc_num/float(acc_num + conf_mat["0"][1]+conf_mat["0"][2]+conf_mat["0"][3]+conf_mat["90"][0]+conf_mat["90"][2]+conf_mat["90"][3]+conf_mat["180"][0]+conf_mat["180"][1]+conf_mat["180"][3]+conf_mat["270"][0]+conf_mat["270"][1]+conf_mat["270"][2]),"%"
	return

# This function will take the training file and create a dictionary with key as "photo id"+'|'+ "orientation" and value as list of rgb vectors 
# for that image
def create_dict(file):
	file_dict=collections.defaultdict(list)
	with open(file) as fl:
		for line in fl:
			lineSplit = line.split()
			file_dict[lineSplit[0]+"|"+lineSplit[1]].append(map(int,lineSplit[2:]))
	for line in file_dict:
		file_dict[line].append(1/float(len(file_dict)))    	        
	return file_dict

# Start of K nearest
# k nearest neighbour algorithm with k=1
def k_nearest(train_dict,test_dict):
	print " Training and testing with k nearest. "
	predict_dict = collections.defaultdict(lambda: 0)
	count = 0
	for test_key in test_dict:
		min_eucldn_dist=infinity
		test_vector=test_dict[test_key]
		original_orient=test_key.split('|')[1]
		for train_key in train_dict:
			curr_eucldn_dist=distance.euclidean(test_vector[0],train_dict[train_key][0])
			if curr_eucldn_dist<min_eucldn_dist:
				min_eucldn_dist=curr_eucldn_dist
				min_key=train_key
		derived_orient=min_key.split('|')[1]
		if derived_orient == "0":
			conf_mat[original_orient][0] += 1
		elif derived_orient == "90":
			conf_mat[original_orient][1] += 1 
		elif derived_orient == "180":
			conf_mat[original_orient][2] += 1 
		elif derived_orient == "270":
			conf_mat[original_orient][3] += 1 
		predict_dict[test_key.split('|')[0]] = derived_orient
	print "confusion matrix is :"
	print conf_mat
	with open ("nearest_output.txt", "w") as fl:
		json.dump(predict_dict, fl, indent = 4)				
	return

# Start of Adaboost 
# This function will create a dictionary of stumps with their corresponding weights.
def create_stump_wt_dict(stump_cnt):
	stump_wt = collections.defaultdict(lambda: 0.0)
	while len(stump_wt) != stump_cnt:
		keys = (rand.randint(0, 191), rand.randint(0, 191))
		if keys[0] != keys [1]:
			stump_wt[keys] = 0	
	return stump_wt

# Creates four stump_wt dictionaries for all four classifiers.
def train_adaboost(stump_cnt, file_dict):
	class_dict = collections.defaultdict(lambda:  collections.defaultdict(lambda: 0.0))
	for key in [0, 90, 180, 270]:
		stump_wt = create_stump_wt_dict(stump_cnt)
		file_dict_copy = json.load( open("train_dict.txt", 'r'))
		class_dict[key] = classifier(key, stump_cnt, stump_wt, file_dict_copy)
	return class_dict	

# Test each test image and return the most probable orientation.
def test_utility(test_image, class_dict):
	orientation = {0:0, 90:0, 180:0, 270:0}
	# Run each classifier on the test image.
	for cls in class_dict:
		weight = 0
		for stumps in class_dict[cls]:
			# This is a stump.
			if test_image[0][stumps[0]] > test_image[0][stumps[1]]:
				weight += class_dict[cls][stumps]
			else:
				weight += -1*class_dict[cls][stumps]	
		orientation[cls] = weight
	# Return the class with maximum stump weight.
	return str(max(orientation, key=orientation.get))

# Test data
def test_adaboost(class_dict, test_dict):
	print " Testing with Adaboost. "
	predict_dict = collections.defaultdict(lambda: 0)
	for image in test_dict:
		original_orient = image.split('|')[1]
		derived_orient = test_utility(test_dict[image], class_dict)
		if derived_orient == "0":
			conf_mat[original_orient][0] += 1
		elif derived_orient == "90":
			conf_mat[original_orient][1] += 1 
		elif derived_orient == "180":
			conf_mat[original_orient][2] += 1 
		elif derived_orient == "270":
			conf_mat[original_orient][3] += 1 	
		predict_dict[image.split('|')[0]]	= derived_orient
	print "The confusion matrix is ", conf_mat
	calc_accuracy(conf_mat)
	return predict_dict

# Train and Test using Adaboost algorithm and create output.txt
def adaboost(stump_cnt, file_dict, test_dict):
	print " Training with Adaboost. "
	class_dict = train_adaboost(stump_cnt, file_dict)
	predict_dict = test_adaboost(class_dict, test_dict)
	with open ("adaboost_output.txt", "w") as fl:
		json.dump(predict_dict, fl, indent = 4)

# Create classifier for each class.
def classifier(class_val, stump_cnt, stump_wt, file_dict):
	for key in stump_wt:
		error = 0.0
		incorrect = 0.0
		correct = 0.0
		for j in file_dict:
			if int (j.split("|")[1]) == class_val:
				if file_dict[j][0][key[0]] > file_dict[j][0][key[1]] :
					correct += 1
				else:
					incorrect += 1
					error += file_dict[j][-1]	
		for j in file_dict:
			if int (j.split("|")[1]) == class_val:
				if file_dict[j][0][key[0]] > file_dict[j][0][key[1]] :
					file_dict[j][-1]  = 0.5 / correct
				else:	

					file_dict[j][-1]  = 0.5 / incorrect	
		if error == 0:
			error = 10e-6	
		if error >= 1:			
			error = 0.99
		stump_wt[key] = 0.5 * math.log((1 - error)/( error))
	return stump_wt


# Start of Neural Net
# Helper function to normalize inpout and output for neural net.
def normalise(file_dict_nnet):
	total =  sum( file_dict_nnet)
	for pixel in range (len(file_dict_nnet)):
		file_dict_nnet[pixel] /= float (total)
		
#Computes derivative of activation function
def derivative(output,act_func='sigmoid'):
	if act_func=='relu':
		return activation_func(output)
	return map(operator.mul,activation_func(output),(map(operator.sub,[1.0]*len(output),activation_func(output))))
	
def activation_func(output,act_func='sigmoid'):
	o1=[]
	for i in range(len(output)):
		if act_func=='sigmoid':
			s=1.0/(1 + math.exp(-output[i]))
			o1.append(s)
		elif act_func=='relu':
			s=math.log(1+math.exp(output[i]))
			o1.append(s)
	return o1

#Computes error for the output layer and hidden layer
def compute_out_err(class_encode,output):
	o1=activation_func(output)
	out_err=map(operator.mul,map(operator.sub, class_encode, o1),derivative(output,'relu'))
	return out_err

def compute_hidden_err(out_err,hid_out_wt,hidden):
	weighted_err=[0.0]*hid_out_wt.shape[1]
	for row in range(len(hid_out_wt)):
		for col in range(len(hid_out_wt[row])):
			weighted_err[col]+=out_err[row]*hid_out_wt[row][col]
	return map(operator.mul,weighted_err,derivative(hidden,'relu'))

# Updates weights
def update_weights(input,wt,err,learning_rate=0.5):
	for row in range(len(wt)):
		for col in range(len(wt[row])):
			wt[row][col]+=(err[row]*learning_rate*input[col])

# Training of neural network.
def train_nnet(hidden_cnt,train_dict,act_func='sigmoid',iterations=5):
	class_encode={0:[1,0,0,0],90:[0,1,0,0],180:[0,0,1,0],270:[0,0,0,1]}
	inp_hid_wt=np.random.random((hidden_cnt, 192))
	hid_out_wt=np.random.random((4,hidden_cnt))
	for key in train_dict:
		normalise(train_dict[key][0])
	for key in train_dict:		
		for it in range(iterations):
			hidden=[]
			for row in inp_hid_wt:
				hidden.append(sum(operator.mul(row,train_dict[key][0])))	
			output=[]
			for wt in hid_out_wt:
				output.append(sum(operator.mul(wt,activation_func(hidden,'relu'))))
			output_err =compute_out_err(class_encode[int(key.split('|')[1])],output)
			hidden_err=compute_hidden_err(output_err,hid_out_wt,hidden)
			update_weights(activation_func(hidden,'relu'),hid_out_wt,output_err)
			update_weights(train_dict[key][0],inp_hid_wt,hidden_err)
	return inp_hid_wt,hid_out_wt

# Neural Net testing.
def test_nnet(test_dict,inp_hid_wt,hid_out_wt,act_func='sigmoid'):
	print " Testing with Neural Net. "
	class_labels={0:0,1:90,2:180,3:270}
	predict_dict = collections.defaultdict(lambda: 0)
	for key in test_dict:	
		normalise(test_dict[key][0])	
		hidden=[]
		original_orient = key.split('|')[1]
		for row in inp_hid_wt:
			hidden.append(sum(operator.mul(row,test_dict[key][0])))		
		output=[]
		for wt in hid_out_wt:
			output.append(sum(operator.mul(wt,activation_func(hidden,'relu'))))
		max_ind=output.index(max(output))
		derived_orient = str(class_labels[max_ind])
		if derived_orient == "0":
			conf_mat[original_orient][0] += 1
		elif derived_orient == "90":
			conf_mat[original_orient][1] += 1 
		elif derived_orient == "180":
			conf_mat[original_orient][2] += 1 
		elif derived_orient == "270":
			conf_mat[original_orient][3] += 1
		predict_dict[key.split('|')[0]] = derived_orient
	print "confusion matrix: ",conf_mat
	calc_accuracy(conf_mat)
	with open ("nnet_output.txt", "w") as fl:
		json.dump(predict_dict, fl, indent = 4)

#neural networks
def nnet(hidden_cnt,train_dict,test_dict):
		print " Training with Neural Net. "
		inp_hid_wt,hid_out_wt=train_nnet(hidden_cnt,train_dict)
		test_nnet(test_dict,inp_hid_wt,hid_out_wt)
		
# MAIN FUNCTION
if __name__ == "__main__": 
	method_var=""   
	[train_file, test_file, method] = sys.argv[1:4] 
	if len(sys.argv)>4:
		   method_var=sys.argv[4]
	train_dict=create_dict(train_file)
	test_dict=create_dict(test_file)
	if method == "nearest":
		k_nearest(train_dict,test_dict)
		calc_accuracy(conf_mat)
	elif method == "adaboost":
		stump_cnt=int(method_var)    
		with open ("train_dict.txt", "w") as fl:
			json.dump(train_dict, fl, indent = 4)
		adaboost(stump_cnt, train_dict, test_dict)
	elif method== "nnet" or method=='best':
		if method=='best':
			hidden_cnt=5
		else:
			hidden_cnt=int(method_var)	
		nnet(hidden_cnt,train_dict,test_dict)
	else:
		print ("Unknown method")
