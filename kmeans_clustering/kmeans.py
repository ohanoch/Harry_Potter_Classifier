import numpy as np
import matplotlib.pyplot as plt
import re
import random
import math
import datetime
import time
import matplotlib.patches as patches
import gmpy2
from sklearn.decomposition import PCA


def encodeToAxis(word_list):
	word_values = [1]
	used_words = []
	for word in word_list:
		if used_words == []:
			used_words.append(word)
			continue

		curr_value = 0
		for i in range(len(used_words)):
			curr_value += word_values[i] * sum_of_words[used_words[i]]

		word_values.append(curr_value + 1)
		used_words.append(word)
	
	for i,value in enumerate(word_values):
		word_values[i] = float(gmpy2.root(value, 750))

	value_dict = {}
	for i in range(len(used_words)):
		value_dict[used_words[i]] = word_values[i]

	return value_dict

def book_to_pages(book, page_length):
	book.seek(0,0)
	page_list = []
	page = ""
	for para_num, para in enumerate(book):
		if para_num % page_length == 0:
			page_list.append(page)
			page = ""
		page += para

	return page_list

def page_value(page):
	point = [0,0]
	for word in re.sub(r'[^\w\s]', '', page.lower()).split():
		if(word[-1] == "s"):
			word = word[:-1]
		if word in x_words:
			point[0] += x_values[word]
		elif word in y_words:
			point[1] += y_values[word]
	return tuple(point)

def pca():
	print "starting pca"
	mat = []
	count = 0
	for pages in book_pages:
		for page in pages:
			count += 1
			page_arr = []
			for word in sum_of_words:
				if word in page:
					page_arr += [sum_of_words[word]]
				else:
					page_arr += [0]

			#print page_arr
			mat += [np.array(page_arr)]
	cov_mat = np.cov(np.array(mat).T)

	print cov_mat.shape
	
	w, v = np.linalg.eig(cov_mat)
	print w
	print w.shape

	plt.hist(np.array(w).real, bins='auto')
	plt.show()

NUM_OF_BOOKS = 7
PARA_PER_PAGE = 80
k = NUM_OF_BOOKS
data_files = [open("../txts/HP" + str(i) + ".txt", "r") for i in range(1,8)]

book_words = [{} for i in range(NUM_OF_BOOKS)]
words_per_book = [0 for i in range(NUM_OF_BOOKS)]

for book_num,book in enumerate(data_files):
	for para in book:
		for letter in para:
			if letter != " " and letter != "\n":
				if letter in "`~!@#$%^&*()_+-=,./\'\"":
					if letter in book_words[book_num]:
						book_words[book_num][letter] += 1
					else:
						book_words[book_num][letter] = 1

		for word in re.sub(r'[^\w\s]', '', para.lower()).split():
			if(word[-1] == "s"):
				word = word[:-1]
			if word in book_words[book_num]:
				book_words[book_num][word] += 1
			else:
				book_words[book_num][word] = 1

			words_per_book[book_num] += 1		

book_words_count = [book_words[i].copy() for i in range(NUM_OF_BOOKS)]
for book_num,words in enumerate(book_words):
	for word in words:
		book_words[book_num][word] /= (words_per_book[book_num]*1.0)


print("before removing: " + str(len(book_words[3])))
thresh = 0.4 * math.pow(10,-5)
for book_num,words in enumerate(book_words):
	temp_dict = {}
	for word in words:
		word_all = [0 for i in range(NUM_OF_BOOKS)]
		word_all[book_num] = words[word]
		for i,book in enumerate(book_words[:book_num]+book_words[book_num+1:]):
			if word in book:	
				if i < book_num:
					word_all[i] = book[word]
				else:
					word_all[i + 1] = book[word]

		num_zeros = 0
		for num in word_all:
			if num == 0:
				num_zeros += 1

		#if np.var(word_all) < thresh:
		if num_zeros < 6:
			temp_dict[word] = 1

		"""
		minVal = words[word]
		maxVal = words[word]
		
		for other_words in book_words[:book_num]+book_words[book_num+1:]:
			if(word in other_words):
				if(other_words[word] < minVal):
					minVal = other_words[word]
				elif(other_words[word] > maxVal):
					maxVal = other_words[word]
			else:
				minVal = thresh * 4
					
		if(minVal / maxVal > thresh):
			temp_dict[word] = 1
		"""
	for word in temp_dict:
		for i in range(len(book_words)):
			if word in book_words[i]:
				del book_words[i][word]
			if word in book_words_count[i]:
				del book_words_count[i][word]

print("after removing: " + str(len(book_words[3])))

sum_of_words = {}
for words in book_words_count:
	for word in words:
		if not word in sum_of_words:
			sum_of_words[word] = words[word]
		else:
			sum_of_words[word] += words[word]

print("total words left from all books " + str(len(sum_of_words)))

#random.seed(time.time())
final_words = sum_of_words.keys()
#random.shuffle(final_words)

x_words = []
y_words = []


x_words = final_words[:len(final_words)/2]
y_words = final_words[len(final_words)/2:]
"""
for i,words in enumerate(book_words):
	for word in words:
		if word in sum_of_words:
			if i <= 4:
				x_words.append(word)
			else:
				y_words.append(word)
"""
x_values = encodeToAxis(x_words)
y_values = encodeToAxis(y_words)

#split books into pages
book_pages = []
for book in data_files:
	book_pages.append(book_to_pages(book, PARA_PER_PAGE))

#get value for each page
page_values = [[] for i in range(NUM_OF_BOOKS)]

for i, pages in enumerate(book_pages):
	print i
	for page in pages:
		value = page_value(page)
		#if value[0] < 1.5 and value[1] < 1.5:
		page_values[i].append(page_value(page))

min_x = float('inf')
max_x = 0
min_y = float('inf')
max_y = 0

for book_values in page_values:
	for value in book_values:
		if value[0] < min_x:
			min_x = value[0]
		if value[0] > max_x:
			max_x = value[0]
		if value[1] < min_y:
			min_y = value[1]
		if value[1] > max_y:
			max_y = value[1]

pca();
"""
pca_func = PCA()
X = []
for i,values in enumerate(page_values):
	for value in values:
		X.append(value)
pca.fit(np.array(X))
print pca._values_
"""
#############################################################################################################
print "starting kmeans"
point_colors = ["cyan","green","red","yellow","grey","magenta","navy"]
center_colors = ["aquamarine", "beige", "lime", "purple", "sienna", "pink", "orange"]

#values of page points
value_x = [[] for i in range(NUM_OF_BOOKS)]
value_y = [[] for i in range(NUM_OF_BOOKS)]
for i,values in enumerate(page_values):
	for value in values:
		value_x[i].append(value[0])
		value_y[i].append(value[1])

centers = [ (random.uniform(min_x, max_x), random.uniform(min_y, max_y)) for i in range(k)]
curr_clusters = [[] for i in range(k)]
is_same = False
while(True):
	old_cluster = curr_clusters
	curr_clusters = [[] for i in range(k)]

	# add points to nearest center - thus defining clusters
	for book_values in page_values:
		for value in book_values:
			curr_distance = float('inf')
			cluster_num = -1
			for i,center in enumerate(centers):
				center_dist = math.sqrt(math.pow(value[0] - center[0],2) + math.pow(value[1] - center[1],2))
				if center_dist < curr_distance:
					curr_distance = center_dist
					cluster_num = i
			curr_clusters[cluster_num].append(value)

	#check for cluster not changing - thus ending process
	is_same = True
	for i,cluster in enumerate(curr_clusters):
		for value in cluster:
			if not value in old_cluster[i]:
				is_same = False	

	if is_same:
		break

	# reinitialise centers
	for i,cluster in enumerate(curr_clusters):
		sum_x = 0
		sum_y = 0
		for value in cluster:
			sum_x += value[0]
			sum_y += value[1]
		if not cluster == []:
			centers[i] = (sum_x / len(cluster), sum_y / len(cluster))
		else:
			print "cluster " + str(i) + " is empty"
	#find radius of clusters
	radiis = [0 for i in range(k)]
	for i,cluster in enumerate(curr_clusters):
		for value in cluster:
			curr_radius = math.sqrt(math.pow(value[0] - centers[i][0],2) + math.pow(value[1] - centers[i][0],2))
			if radiis[i] < curr_radius:
				radiis[i] = curr_radius

	print "plotting graph..."
	print "time is: " + str(datetime.datetime.now())
	#color=iter(plt.cm.rainbow(np.linspace(0,1,k*2)))
	for i,center in enumerate(centers):
		#c = next(color)
		plt.plot([center[0]],[center[1]], "x", color=center_colors[i])
		plt.Circle((center[0], center[1]), radiis[i], facecolor=center_colors[i], alpha=0.4)

	#	for value in curr_clusters[i]:
	#		plt.plot([value[0]],[value[1]], ".", color=c)
			

	for i,values in enumerate(page_values):
		#c = next(color)
		plt.plot(value_x[i], value_y[i], ".", color=point_colors[i])

	plt.show()
