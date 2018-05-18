import numpy as np
import math
import random

word_dic = {}
pages = []

def readInData():
	word_index = 0

	for i in range(1,8):	#Reads in books 1 to 7
		file_directory = 'hp_books/HP'+str(i)+'.txt'
		print "Reading in from: " + file_directory
		book_file  = open(file_directory, 'r')
		in_pages = book_file.readlines()			#Reads in all lines of a file and returns them as a page
		split_pages = []
		for page in in_pages:
			#Goes through each word and removes unnecessary punctuation at the end of words such as "..." and also puts spaced
			#before and after question marks and similar puntuation so they form their own words when split. The splits
			#the pages (a page is a single string) into words and adds these split pages into the split_pages array
			page = page.replace(",", "").replace(".", "").replace("?", " ? ").replace("!", " ! ").replace("\"", 			"").replace(":","").replace(";", "").replace("\n", "").replace(")", "").replace("(", "").replace("'", 				"").replace("\xe2\x80\x99","'").replace("\xe2\x80\x9d", "").replace("\xe2\x80\x9d", "").replace("\xe2\x80\x9c", 			"").replace("\xe2\x80\x94","")
			pages.append([page, i])
			split_pages.append(page.split(" "))
		#print pages_and_class
	
		#Adds the words to the word dictionary. A word should only be added once
		for page in split_pages:
			for word in page:
				if not word in word_dic:
					word_dic[word] = word_index
					word_index += 1
		#return pages_and_class
	
#Creates a weight matrix of the correct dimensions. First param is the size you want the layer, second param is the layer
#which will input to this layer, used here for the dimensions of the weight matrix
def initNetworkLayer(size, input_layer, variance):
	return(np.random.normal(0, variance, [size, input_layer.shape[0]]))

#Activation funtions
def sigmoid(z_array):
	#print "Z array is: "+str(z_array)
	#print "*******************************************************************************************************"
	sigOut = np.zeros(z_array.shape[0])
	for i in range(z_array.shape[0]):
		sigOut[i] = 1/(1+math.exp(-z_array[i]))
	return sigOut


def ReLU(z_array):
	sigOut = np.zeros(z_array.shape[0])
	for i in range(z_array.shape[0]):
		if z_array[i] > 0:
			sigOut[i] = z_array[i]
		else:
			sigOut[i] = 0
	return sigOut

def ReLU_derivative(activation_array):
	derivatives_array = np.zeros(len(activation_array))
	for i in range(len(activation_array)):
		if activation_array[i] > 0:
			derivatives_array[i] = 1
	return derivatives_array

#Computes the activation for the layer which weights are given. The first parameter is the output of the layer before it
def computeLayerActivation(inputLayer, currentLayerWeights):
	return(sigmoid(np.matmul(currentLayerWeights, inputLayer)))

def main():
	#Hyper-parameters
	learning_rate = 0.2
	regularization_rate = 0.0
	depth = 3
	training_percent = 0.6
	validation_percent = 0.2
	test_percent = 0.2

	readInData()
	random.shuffle(pages)

	#Splits pages into training, validation and test data sets
	training_pages = pages[:int(training_percent*len(pages))]
	validation_pages = pages[int(training_percent*len(pages)): int((training_percent+validation_percent)*len(pages))]
	test_pages = pages[int((training_percent+validation_percent)*len(pages)): int((training_percent		+validation_percent	+test_percent)*len(pages))]
	
	#Just used to get the dimensions for the input
	input_dim_holder = np.zeros(len(word_dic))
	input_dim_holder = np.transpose(input_dim_holder)

	#Defining architecture. The first parameter is the size of the layer, the second is the layer before it
	w1 = initNetworkLayer(200, input_dim_holder, 0.15)
	w2 = initNetworkLayer(20, w1, 0.2)
	w3 = initNetworkLayer(7, w2, 0.5)

	#print w1
	#print "###########################################################################################################"
	#print w2
	#print "***********************************************************************************************************"
	#print w3
	#print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
	
	#Defines update matrices for each weight
	w1_update = initNetworkLayer(200, input_dim_holder, 0)
	w2_update = initNetworkLayer(20, w1, 0)
	w3_update = initNetworkLayer(7, w2, 0)

	num_correct = 0

	#out_error_sum = np.zeros(7)
	batch_error = 0
	for page in training_pages:
		input_array = np.zeros(len(word_dic))
		for word in page[0].split(" "):						#page[0] is the actual page, page[1] is the book number its from
			input_array[word_dic[word]] = 1
		input_array_transposed = np.transpose(input_array)
	
		a1 = computeLayerActivation(input_array_transposed, w1)
		a2 = computeLayerActivation(a1, w2)
		a3 = computeLayerActivation(a2, w3)
		classification = np.argmax(a3)

		#print a1
		#print "###########################################################################################################"
		#print a2
		#print "***********************************************************************************************************"
		#print a3
		#print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
		
		#Computing loss
		y = np.zeros(7)
		y[page[1]-1] = 1
		#print "Correct: "+str(y)+" Actual index: "+str(classification)+" Output: "+str(a3)
		log_output = np.zeros(7)
		for i in range(a3.shape[0]):
			log_output[i] = math.log(a3[i])
		error = np.dot(y, log_output)+np.dot((1-y), (1-log_output))
		batch_error += error


		#print "Classified class: "+str(classification)+". Actual is: "+str(page[1]-1)
		#print str(a3) + "              "+ str(y)
		#	if classification == (page[1]-1):
			#print "                                                                           Correct"

		#Computing back prop values
		a3_error = np.multiply((a3 - y),np.multiply(a3,(1-a3)))					#This is for sigmoid activation
		#a3_error = np.multiply((a3 - y), ReLU_derivative(a3))
		a2_error = np.multiply(np.dot(np.transpose(w3), a3_error),np.multiply(a2,(1-a2))) 
		a1_error = np.multiply(np.dot(np.transpose(w2), a2_error),np.multiply(a1,(1-a1)))

		partial_derive_w1 = np.dot(a1_error.reshape(200,1), input_array.reshape(1,len(input_array)))
		partial_derive_w2 = np.dot(a2_error.reshape(20,1), a1.reshape(1,200))
		partial_derive_w3 = np.dot(a3_error.reshape(7,1), a2.reshape(1,20))
		
		print partial_derive_w1
		print "#######################################################################################################"
		print partial_derive_w2
		print "*******************************************************************************************************"
		print partial_derive_w3
		print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"

		#Update weights with gradient descent
		w1 -= learning_rate*partial_derive_w1
		w2 -= learning_rate*partial_derive_w2
		w3 -= learning_rate*partial_derive_w3 
		
		print "Updated weights"

	
	regularizer = np.sum(np.power(w1,2)) + np.sum(np.power(w2,2)) + np.sum(np.power(w3,2))
	#j = (-1/len(page) * batch_error) + ((regularization_rate/(2*len(page)))*regularizer)

	



#w1_update += input_array*(np.linalg.norm(a3 - y))
		#w2_update += a1*(np.linalg.norm(a3 - y))
		#w3_update += a2*(np.linalg.norm(a3 - y))
if __name__ == '__main__':
    main()
