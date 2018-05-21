import numpy as np
import math
import random
import logging
import datetime
import time

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
	
		#Adds the words to the word dictionary. A word should only be added once
		for page in split_pages:
			for word in page:
				if not word in word_dic:
					word_dic[word] = word_index
					word_index += 1

		
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

def hyper_tan(activation_array):
	return np.tanh(activation_array)

def softmax(z_array):
	summation = sum(np.exp(z_array))
	return_array = np.zeros(len(z_array))
	for i in range(len(z_array)):
		return_array[i] = np.exp(z_array[i])/summation
	return return_array

def main():
	#Hyper-parameters
	learning_rate = 0.1		#Starting learning rate
	regularization_rate = 0.01	#Starting regularization rate
	training_percent = 0.6
	validation_percent = 0.2
	test_percent = 0.2
	num_batches = 50				#Starting number of batches
	num_epochs = 50
	best_validation = 0
	best_validation_index = 0

	readInData()

	evened_pages = pages[0:347] + pages[0:347] + pages[0:347] + pages[347: 724] + pages[347: 724] + pages[347: 724] + pages[724: 1210] + pages[724: 1210]  + pages[724: 924] + pages[1210: 2019] + pages[1400:1600] + pages[2019: 3123] + pages[3123:3851] + pages[3200:3500] + pages[3851:4700] + pages[3900:4050]
	
	random.shuffle(evened_pages)

	#Splits pages into training, validation and test data sets
	training_pages = evened_pages[:int(training_percent*len(evened_pages))]
	validation_pages = evened_pages[int(training_percent*len(evened_pages)): int((training_percent+validation_percent)*len(evened_pages))]
	test_pages = evened_pages[int((training_percent+validation_percent)*len(evened_pages)): int((training_percent		+validation_percent	+test_percent)*len(evened_pages))]
	
	#Just used to get the dimensions for the input
	input_dim_holder = np.zeros(len(word_dic))
	input_dim_holder = np.transpose(input_dim_holder)

	
	num_correct = 0
	for j in range(11):					#Validation Iterations

		#Defining architecture. The first parameter is the size of the layer, the second is the layer before it
		wS = initNetworkLayer(7, input_dim_holder, 0.01)

		wS_update = initNetworkLayer(7, input_dim_holder, 0.0)

		if j == 10:
			j = best_validation_index

		#num_batches += (j*10)
		learning_rate += j/10
		#regularization_rate += j/10
		for i in range(num_epochs):
			batch_size = len(training_pages)/num_batches
			for i in range(1, num_batches):
				batch_pages = training_pages[batch_size*(i-1): batch_size*i]
				random.seed(time.time())
				random.shuffle(batch_pages)
				batch_error = 0
				batch_correct = 0.0
				partial_derive_wS = 0
				for page in batch_pages:
					input_array = np.zeros(len(word_dic))
					for word in page[0].split(" "):			#page[0] is the actual page, page[1] is the book number its from
						if word in word_dic:
							input_array[word_dic[word]] = 1
					input_array_transposed = np.transpose(input_array)
	
					#Feed forward
					zS = np.matmul(wS, input_array_transposed)
					aS = softmax(zS)
					classification = np.argmax(aS)

					#Computing loss
					y = np.zeros(7)
					y[page[1]-1] = 1
					
					if classification == (page[1]-1):
						batch_correct += 1.0
					#	print "                                                                           Correct"

					#Computing layer errors from back prop 
					aS_error = aS-y

					#Computing weight updates
					partial_derive_wS += np.dot(aS_error.reshape(7,1), input_array.reshape(1,len(input_array)))


				#Update weights with gradient descent
				wS -= (learning_rate/batch_size)*partial_derive_wS + (regularization_rate*wS)
								
			print "End epoch"
		print "End train"

		confusion_matrix = np.zeros([7,7])
		for page in validation_pages:
			input_array = np.zeros(len(word_dic))
			for word in page[0].split(" "):				#page[0] is the actual page, page[1] is the book number its from
				if word in word_dic:
					input_array[word_dic[word]] = 1
			input_array_transposed = np.transpose(input_array)

			#Feed forward
			zS = np.matmul(wS, input_array_transposed)
			aS = softmax(zS)
			classification = np.argmax(aS)

			confusion_matrix[classification][page[1]-1] += 1
		accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3] + confusion_matrix[4][4] + confusion_matrix[5][5] + confusion_matrix[6][6])/len(validation_pages)
		if accuracy > best_validation:
			best_validation_index = j
			best_validation = accuracy
		#print "Validation Accuracy: " + str(accuracy)
		#print confusion_matrix

		logging.basicConfig(level=logging.DEBUG,
				            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
				            datefmt='%m-%d %H:%M',
				            filename='./validation'+str(datetime.datetime.now())+'.log',
				            filemode='w')
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
		console.setFormatter(formatter)
		logging.getLogger('').addHandler(console)
		logging.info("Validation Accuracy: " + str(accuracy))
		logging.info(confusion_matrix)


	print best_validation_index
	print best_validation
	confusion_matrix = np.zeros([7,7])
	for page in test_pages:
		input_array = np.zeros(len(word_dic))
		for word in page[0].split(" "):				#page[0] is the actual page, page[1] is the book number its from
			if word in word_dic:
				input_array[word_dic[word]] = 1
		input_array_transposed = np.transpose(input_array)

		#Feed forward
		zS = np.matmul(wS, input_array_transposed)
		aS = softmax(zS)
		classification = np.argmax(aS)

		confusion_matrix[classification][page[1]-1] += 1
	accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3] + confusion_matrix[4][4] + confusion_matrix[5][5] + confusion_matrix[6][6])/len(validation_pages)
	if accuracy > best_validation:
		best_validation_index = j
		best_validation = accuracy
	print "Test Accuracy: " + str(accuracy)
	print confusion_matrix

	
if __name__ == '__main__':
    main()
