import re

def encodeToAxis(word_list):
	word_values = [1]
	used_words = []
	for word in word_list:
		if used_words == []:
			used_words.append(word)
			continue

		curr_value = 1
		for i in range(len(used_words)):
			curr_value += word_values[i-1] * sum_of_words[used_words[i]]

		word_values.append(curr_value)
		used_words.append(word)

	value_dict = {}
	for i in range(len(used_words)):
		value_dict[used_words[i]] = word_values[i]

	return value_dict

data_files = [open("../txts/HP" + str(i) + ".txt", "r") for i in range(1,8)]

book_words = [{} for i in range(7)]
words_per_book = [0 for i in range(7)]

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

#book_words_count = [book_words[i].copy() for i in range(7)]
for book_num,words in enumerate(book_words):
	for word in words:
		book_words[book_num][word] /= (words_per_book[book_num]*1.0)


print("before removing: " + str(len(book_words[0])))
thresh = 0.07
for book_num,words in enumerate(book_words):
	temp_dict = {}
	for word in words:
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
	for word in temp_dict:
		for i in range(len(book_words)):
			if word in book_words[i]:
				del book_words[i][word]
	#		if word in book_words_count[i]:
	#			del book_words_count[i][word]

print("after removing: " + str(len(book_words[0])))

sum_of_words = {}
for words in book_words:
	for word in words:
		if not word in sum_of_words:
			sum_of_words[word] = words[word]
		else:
			sum_of_words[word] += words[word]

print("total words left from all books " + str(len(sum_of_words)))

#print(sum_of_words)
print(encodeToAxis(sum_of_words))
