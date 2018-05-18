import numpy as np
import matplotlib.pyplot as plt
import re
import random
import math
import datetime
import time
import matplotlib.patches as patches
import gmpy2
import logging
from sklearn.decomposition import PCA

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/kmeans_devon_' + str(datetime.datetime.now()) + '.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

def remove_words():
	logging.info("before removing: " + str(len(book_words[3])))
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
			if num_zeros < 6 or (num_zeros == 6 and words[word] < thresh):
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

	logging.info("after removing: " + str(len(book_words[3])))

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

def page_value(page, book_num):
	"""
	point = [0,0]
	for word in re.sub(r'[^\w\s]', '', page.lower()).split():
		if(word[-1] == "s"):
			word = word[:-1]
		if word in x_words:
			point[0] += x_values[word]
		elif word in y_words:
			point[1] += y_values[word]
	return tuple(point)
	"""
	p_value = []
	for key in sum_of_words.keys():
		p_value.append(page.count(key) / (words_per_book[book_num]*1.0))
			
	return p_value

def pca():
	logging.info("starting pca")
	"""
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
	"""
	mat = [np.array(value) for book_values in page_values for value in book_values]
	cov_mat = np.cov(np.array(mat).T)
		

	logging.info(cov_mat.shape)
	
	w, v = np.linalg.eig(cov_mat)
	small_w = [a for a in np.array(w).real if a != 0]
	logging.info(np.array(small_w))
	logging.info(np.array(small_w).shape)

	logging.info("plotting pca...")
	plt.plot(range(len(small_w)), small_w, "ro")
	#plt.hist(small_w, bins='auto')
	plt.show()

NUM_OF_BOOKS = 7
PARA_PER_PAGE = 5
k = 37#NUM_OF_BOOKS
do_pca = False
random.seed(time.time())
logging.info("NUM_OF_BOOKS: " + str(NUM_OF_BOOKS) )
logging.info("PARA_PER_PAGE: " + str(PARA_PER_PAGE) )
logging.info("k: " + str(k) )

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


remove_words();

sum_of_words = {}
for words in book_words_count:
	for word in words:
		if not word in sum_of_words:
			sum_of_words[word] = words[word]
		else:
			sum_of_words[word] += words[word]

logging.info("total words left from all books " + str(len(sum_of_words)))
"""
#random.seed(time.time())
final_words = sum_of_words.keys()
#random.shuffle(final_words)

x_words = []
y_words = []


x_words = final_words[:len(final_words)/2]
y_words = final_words[len(final_words)/2:]
"""
"""
for i,words in enumerate(book_words):
	for word in words:
		if word in sum_of_words:
			if i <= 4:
				x_words.append(word)
			else:
				y_words.append(word)
"""
"""
x_values = encodeToAxis(x_words)
y_values = encodeToAxis(y_words)
"""

#split books into pages
book_pages = []
for book in data_files:
	book_pages.append(book_to_pages(book, PARA_PER_PAGE))
logging.info("amount of pages: " + str(len([page for book in book_pages for page in book])))

#get value for each page
page_values = [[] for i in range(NUM_OF_BOOKS)]

for book_num, pages in enumerate(book_pages):
	logging.info(book_num)
	for page in pages:
		#value = page_value(page)
		page_values[book_num].append(page_value(page, book_num))
"""
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
"""
if do_pca:
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
logging.info("starting kmeans")
point_colors = ["cyan","green","red","yellow","grey","magenta","navy"]
center_colors = ["aquamarine", "beige", "lime", "purple", "sienna", "pink", "orange"]

"""
#values of page points
value_x = [[] for i in range(NUM_OF_BOOKS)]
value_y = [[] for i in range(NUM_OF_BOOKS)]
for i,values in enumerate(page_values):
	for value in values:
		value_x[i].append(value[0])
		value_y[i].append(value[1])
"""
logging.info("initializing centers")
centers = [[random.uniform( \
				(max([page_value[j] for book_values in page_values for page_value in book_values]) + min([page_value[j] for book_values in page_values for page_value in book_values]))/2,\
				3 * (max([page_value[j] for book_values in page_values for page_value in book_values]) + min([page_value[j] for book_values in page_values for page_value in book_values]))/4\
			) for j in range(len(sum_of_words))] for i in range(k)]

logging.info("centers initialized")
curr_clusters = [[] for i in range(k)]
is_same = False
epoch = 0
while(True):
	logging.info("starting epoch " + str(epoch))
	old_clusters = curr_clusters
	curr_clusters = [[] for i in range(k)]

	# add points to nearest center - thus defining clusters
	for book_num, book_values in enumerate(page_values):
		for value in book_values:
			curr_distance = float('inf')
			cluster_num = -1
			for i,center in enumerate(centers):
				center_dist = 0
				for value_dim, center_dim in zip(value, center):
					center_dist += math.pow(value_dim - center_dim,2)
				center_dist = math.sqrt(center_dist)
				if center_dist < curr_distance:
					curr_distance = center_dist
					cluster_num = i
			curr_clusters[cluster_num].append((value, book_num))

	#check for cluster not changing - thus ending process
	is_same = True
	for cluster_num,(old_cluster,cluster) in enumerate(zip(old_clusters, curr_clusters)):
		for value in cluster:
			old_books_values = [val[0] for val in old_cluster]
			if not value[0] in old_books_values:
				#print "cluster that has changed: " + str(cluster_num)
				is_same = False	

	if is_same:
		break

	# reinitialise centers
	for num,cluster in enumerate(curr_clusters):
		"""
		sum_x = 0
		sum_y = 0
		for value in cluster:
			sum_x += value[0]
			sum_y += value[1]
		"""
		cluster_sum = [sum([page[0][i] for page in cluster]) for i in range(len(sum_of_words))]
		if not cluster == []:
			centers[num] = [val / len(cluster) for val in cluster_sum]
		else:
			logging.info("cluster " + str(num) + " is empty")
	
	logging.info("---------------------- epoch " + str(epoch) + " ----------------------------")
	for cluster_num, cluster in enumerate(curr_clusters):
		book_count = [0 for i in range(NUM_OF_BOOKS)]
		for page in cluster:
			book_count[page[1]] += 1
		logging.info("cluster " + str(cluster_num) + " book_count: " + str(book_count))

	epoch += 1
		
	"""
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
	"""
