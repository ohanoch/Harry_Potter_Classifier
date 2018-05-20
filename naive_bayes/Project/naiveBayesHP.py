import random
import math

# Meir Rosendorff

pageSizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]#, 11, 12, 13]
startPoint = [2, 2, 3, 3, 5, 5, 5, 5.5, 6, 6, 6.5, 7, 7]
for pageSize in pageSizes:
    # bookLines = []
    numBooks = 7
    TrainingPercent = 60
    bookPages = []
    testData = []
    trainingData = []
    validationData = []
    bestWords = []
    bestUpperThreshold = 0.0
    bestLowerThreshold = 0.0
    bestNumNeccesary = 0
    globalBestCorrect = 0
    upperThreshold = 2*startPoint[pageSize -1] / 100.0
    lowerThreshold = startPoint[pageSize -1] / 100.0
    alpha = 0.001*pageSize

    for i in range(numBooks):
        bookFile = open("BenchmarkBooks/book" + str(i+1) + ".txt", "r")
        currBook = []
        line = bookFile.readline()
        count = 0
        currLine = str(i) + " "
        lineNum = 0
        while line:
            if line == "" or line == "\n":
                line = bookFile.readline()
                continue
    # get rid of all punction except ! and ?, but seperate ! and ? from the word they are next to
    # to make them seperate words
            line = line.replace(",", "").replace(".", "").replace("?", " ? ").replace("!", " ! ").replace("\"", "").replace(":", "").replace(";", "").replace("\n", "")
            line = line.replace(")", "").replace("(", "").replace("'", "")
            currLine = currLine + " " + line
            lineNum += 1
            if lineNum ==  pageSize:
                currLine = currLine.lower()
                wordArray = currLine.split(" ")
                currBook.append(wordArray)
                count+=1
                lineNum = 0
                currLine = str(i) + " "
            line = bookFile.readline()

        wordArray = currLine.split(" ")
        currBook.append(wordArray)
        count += 1
        bookPages.append(count)
        bookFile.close()

        random.Random(42).shuffle(currBook)
        allLines = currBook[:]
        numCases = len(currBook)*TrainingPercent/100
        numTests = (len(currBook) - numCases)/2

        dataHolder = currBook[numCases:]
        testData = testData + dataHolder[numTests:]
        validationData = validationData + dataHolder[:numTests]

        trainingData = trainingData + currBook[:numCases]

    numTrainingCases = len(trainingData)
    numValidationCases = len(validationData)
    numTestCases = len(testData)
    print "Num Training Cases: " + str(numTrainingCases)
    print "Num Testing Cases:\t" + str(numTestCases)

    individualProbabilities = [0]*numBooks

    #Calculate the probability of it being any one example
    for i in range(numBooks):
        individualProbabilities[i] = bookPages[i]/float(numTrainingCases)
    print bookPages
    words = []
    probabilityTable = []

    # # get the probability of each word for each book
    for i in range(len(trainingData)):
        currPage = trainingData[i]
        bookNum = int(trainingData[i][0])
        print "Page " + str(i+1) + " of " + str(numTrainingCases)
        for j in range(1,len(currPage),1): #start at 1 as pos 0 is the bookNumber
            numPerBook = [0.0]*numBooks #array to count how many times word appears per book
            word = currPage[j] #iterate over the words
            if word in words or word == "": #if the word is blank or its already been seen skip
                continue
            else:
                numPerBook[bookNum] += 1 #add one for the word we just read
    
            for k in range(i+1, len(trainingData), 1): #iterate through the remaining pages checking for the word
                if word in trainingData[k]:
                    currNum = int(trainingData[k][0])
                    numPerBook[currNum]+=1
    
            words.append(word) # add the word to my word list
            probabilities = [0.0]*numBooks
            for k in range(numBooks):# calculate the probability for that word
                probabilities[k] = (numPerBook[k]+1) / (bookPages[k] + numBooks)
            probabilityTable.append(probabilities)
            
    # Save Probabilities to probabilities.txt
    # format is word P(book1) P(book2) ...
    
    probabiltyFile = open("probabilitiesPS" + str(pageSize) + ".txt", "w")
    
    for i in range(len(words)):
        line = words[i]
        for j in range(numBooks):
            line = line + " " + str(probabilityTable[i][j])
        line = line + "\n"

        probabiltyFile.write(line)
 #  	probabiltyFile.close()
	'''
    # # read in from probability files

    probFile = open("probabilitiesPS" + str(pageSize) + ".txt", "r")
    words = []
    probabilityTable = []
    line = probFile.readline()

    while line:

        probs = line.split(" ")
        if not (len(probs)  == numBooks):
        	continue
        
        print probs[0]	
        words.append(probs[0])
        probabilities = []

        for i in range(1, numBooks+1, 1):
            probabilities.append(float(probs[i]))

        probabilityTable.append(probabilities)
        line = probFile.readline()

    #eliminate book numbers
	
    for i in range(numBooks):
        pos = words.index(str(i))
        words.pop(pos)
        probabilityTable.pop(pos)
	'''
    for numNeccesary in [1, 2, 3, 4, 5]:
        print "Num Necesary\t" + str(numNeccesary)
        # learn hyperparameters
        overalImproving = True
        overallPrevBest = 0
        prevLower = 0
        prevUpper = 0
        upperThreshold = 2 * startPoint[pageSize - 1] / 100.0
        lowerThreshold = startPoint[pageSize - 1] / 100.0
        alpha = 0.001 * pageSize
        while(overalImproving):
            improving = True
            prevBest = 0
            while(improving):

                toDelete = []
                for i in range(len(words)):
                    suffcientSize = False
                    timesLow = 0
                    for j in range(len(probabilityTable[i])):
                        if probabilityTable[i][j] >= upperThreshold:
                            suffcientSize = True
                        elif probabilityTable[i][j] < lowerThreshold:
                            timesLow += 1
                    if not suffcientSize or timesLow < numNeccesary:
                        toDelete.append(i)

                currWords = words[:]
                currProbabilityTable = probabilityTable[:]

                print "total Deleted: " + str(len(toDelete))
                totalPopped = 0
                for i in toDelete:
                    i = i - totalPopped
                    currWords.pop(i)
                    currProbabilityTable.pop(i)
                    totalPopped += 1

                print "Num words Reamining:\t" + str(len(currWords))
                # Testing Time
                confusionMatrix = [[0]*numBooks for i in range(numBooks)]

                pageNum = 0
                numCorrect = 0.0
                numIncorrect = 0.0
                for page in validationData:

                    # print "Testing Page Number: " + str(pageNum) + "\tof\t" + str(numTestCases)

                    pageNum += 1

                    caseProbabilities = [1] * numBooks

                    for i in range(len(currWords)):
                        currWord = currWords[i]

                        included = 0
                        if currWord in page:
                            included = 1

                        for j in range(numBooks):
                            prob = currProbabilityTable[i][j]
                            caseProbabilities[j] += math.log(((0.0+included)*prob) + ((1.0-included)*(1.0-prob)))

                    posteriorProbabilities = []
                    for i in range(numBooks):
                        posteriorProbabilities.append(caseProbabilities[i]+math.log(individualProbabilities[i]))

                    probabilityOfAll = sum(posteriorProbabilities)
                    for i in range(numBooks):
                        posteriorProbabilities[i] -= probabilityOfAll


                    max = posteriorProbabilities[0]
                    maxPos = 0
                    for i in range(numBooks):
                        if max < posteriorProbabilities[i]:
                            maxPos = i
                            max = posteriorProbabilities[i]
                    if maxPos == int(page[0]):
                        numCorrect+=  1
                    else:
                        numIncorrect += 1


                    confusionMatrix[maxPos][int(page[0])] += 1

                bestCorrect = numCorrect/numTestCases
                if (bestCorrect >= prevBest):
                    prevBest = bestCorrect
                    lowerThreshold = lowerThreshold - alpha
                    if lowerThreshold <= 0:
                        lowerThreshold = (startPoint[pageSize - 1] + 1) / 100.0
                        improving = False
                    print "L\t" + str(lowerThreshold)
                else:
                    improving = False
                    lowerThreshold += alpha

            improving = True
            prevBest = 0
            while (improving):

                toDelete = []
                for i in range(len(words)):
                    suffcientSize = False
                    timesLow = 0
                    for j in range(len(probabilityTable[i])):
                        if probabilityTable[i][j] >= upperThreshold:
                            suffcientSize = True
                        elif probabilityTable[i][j] < lowerThreshold:
                            timesLow += 1
                    if not suffcientSize or timesLow < numNeccesary:
                        toDelete.append(i)

                currWords = words[:]
                currProbabilityTable = probabilityTable[:]

                print "total Deleted: " + str(len(toDelete))
                totalPopped = 0
                for i in toDelete:
                    i = i - totalPopped
                    currWords.pop(i)
                    currProbabilityTable.pop(i)
                    totalPopped += 1

                print "Num words Reamining:\t" + str(len(currWords))
                # Testing Time
                confusionMatrix = [[0] * numBooks for i in range(numBooks)]

                pageNum = 0
                numCorrect = 0.0
                numIncorrect = 0.0
                for page in validationData:

                    # print "Testing Page Number: " + str(pageNum) + "\tof\t" + str(numTestCases)

                    pageNum += 1

                    caseProbabilities = [1] * numBooks

                    for i in range(len(currWords)):
                        currWord = currWords[i]

                        included = 0
                        if currWord in page:
                            included = 1

                        for j in range(numBooks):
                            prob = currProbabilityTable[i][j]
                            caseProbabilities[j] += math.log(((0.0 + included) * prob) + ((1.0 - included) * (1.0 - prob)))

                    posteriorProbabilities = []
                    for i in range(numBooks):
                        posteriorProbabilities.append(caseProbabilities[i] + math.log(individualProbabilities[i]))

                    probabilityOfAll = sum(posteriorProbabilities)
                    for i in range(numBooks):
                        posteriorProbabilities[i] -= probabilityOfAll

                    max = posteriorProbabilities[0]
                    maxPos = 0
                    for i in range(numBooks):
                        if max < posteriorProbabilities[i]:
                            maxPos = i
                            max = posteriorProbabilities[i]
                    if maxPos == int(page[0]):
                        numCorrect += 1
                    else:
                        numIncorrect += 1
                    confusionMatrix[maxPos][int(page[0])] += 1

                bestCorrect = numCorrect / numTestCases
                if (bestCorrect >= prevBest):
                    prevBest = bestCorrect
                    upperThreshold += 2*alpha
                    print "U\t" + str(upperThreshold)
                else:
                    improving = False
                    upperThreshold -= alpha

            toDelete = []
            for i in range(len(words)):
                suffcientSize = False
                timesLow = 0
                for j in range(len(probabilityTable[i])):
                    if probabilityTable[i][j] >= upperThreshold:
                        suffcientSize = True
                    elif probabilityTable[i][j] < lowerThreshold:
                        timesLow += 1
                if not suffcientSize or timesLow < numNeccesary:
                    toDelete.append(i)

            currWords = words[:]
            currProbabilityTable = probabilityTable[:]

            print "total Deleted: " + str(len(toDelete))
            totalPopped = 0
            for i in toDelete:
                i = i - totalPopped
                currWords.pop(i)
                currProbabilityTable.pop(i)
                totalPopped += 1

            print "Num words Reamining:\t" + str(len(currWords))
            # Testing Time
            confusionMatrix = [[0] * numBooks for i in range(numBooks)]

            pageNum = 0
            numCorrect = 0.0
            numBufferCorrect = 0.0
            numIncorrect = 0.0
            for page in validationData:

                # print "Testing Page Number: " + str(pageNum) + "\tof\t" + str(numTestCases)

                pageNum += 1

                caseProbabilities = [1] * numBooks

                for i in range(len(currWords)):
                    currWord = currWords[i]

                    included = 0
                    if currWord in page:
                        included = 1

                    for j in range(numBooks):
                        prob = currProbabilityTable[i][j]
                        caseProbabilities[j] += math.log(((0.0 + included) * prob) + ((1.0 - included) * (1.0 - prob)))

                posteriorProbabilities = []
                for i in range(numBooks):
                    posteriorProbabilities.append(caseProbabilities[i] + math.log(individualProbabilities[i]))

                probabilityOfAll = sum(posteriorProbabilities)
                for i in range(numBooks):
                    posteriorProbabilities[i] -= probabilityOfAll

                max = posteriorProbabilities[0]
                maxPos = 0
                for i in range(numBooks):
                    if max < posteriorProbabilities[i]:
                        maxPos = i
                        max = posteriorProbabilities[i]
                if maxPos == int(page[0]):
                    numCorrect += 1
                else:
                    numIncorrect += 1
                if abs(maxPos - int(page[0])) <= 1:
                    numBufferCorrect += 1
                confusionMatrix[maxPos][int(page[0])] += 1

            bestCorrect = numCorrect / numTestCases
            if (bestCorrect >= overallPrevBest and alpha > 0.00001):
                overallPrevBest = bestCorrect
                prevLower = lowerThreshold
                prevUpper = upperThreshold
                lowerThreshold += alpha
                upperThreshold -= 2*alpha
                alpha = alpha/2
            else:
                overalImproving = False
                lowerThreshold = prevLower
                upperThreshold = prevUpper
                validationBest = overallPrevBest
            print "Percentage Correct:\t" + str(numCorrect/numTestCases*100)
            print "Percentage Incorrect:\t" + str(numIncorrect / numTestCases * 100)

        if validationBest > globalBestCorrect:
            bestUpperThreshold = prevUpper
            bestLowerThreshold = prevLower
            bestNumNeccesary = numNeccesary
            globalBestCorrect = validationBest

    validationBest = globalBestCorrect
    upperThreshold = bestUpperThreshold
    lowerThreshold = bestLowerThreshold
    numNeccesary = bestNumNeccesary
    # Tests on testing data
    toDelete = []
    for i in range(len(words)):
        suffcientSize = False
        timesLow = 0
        for j in range(len(probabilityTable[i])):
            if probabilityTable[i][j] >= upperThreshold:
                suffcientSize = True
            elif probabilityTable[i][j] < lowerThreshold:
                timesLow += 1
        if not suffcientSize or timesLow < numNeccesary:
            toDelete.append(i)

    currWords = words[:]
    currProbabilityTable = probabilityTable[:]

    print "total Deleted: " + str(len(toDelete))
    totalPopped = 0
    for i in toDelete:
        i = i - totalPopped
        currWords.pop(i)
        currProbabilityTable.pop(i)
        totalPopped += 1

    print "Num words Reamining:\t" + str(len(currWords))
    # Testing Time
    confusionMatrix = [[0] * numBooks for i in range(numBooks)]

    pageNum = 0
    numCorrect = 0.0
    numBufferCorrect = 0.0
    numIncorrect = 0.0
    for page in testData:

        # print "Testing Page Number: " + str(pageNum) + "\tof\t" + str(numTestCases)

        pageNum += 1

        caseProbabilities = [1] * numBooks

        for i in range(len(currWords)):
            currWord = currWords[i]

            included = 0
            if currWord in page:
                included = 1

            for j in range(numBooks):
                prob = currProbabilityTable[i][j]
                caseProbabilities[j] += math.log(((0.0 + included) * prob) + ((1.0 - included) * (1.0 - prob)))

        posteriorProbabilities = []
        for i in range(numBooks):
            posteriorProbabilities.append(caseProbabilities[i] + math.log(individualProbabilities[i]))

        probabilityOfAll = sum(posteriorProbabilities)
        for i in range(numBooks):
            posteriorProbabilities[i] -= probabilityOfAll

        max = posteriorProbabilities[0]
        maxPos = 0
        for i in range(numBooks):
            if max < posteriorProbabilities[i]:
                maxPos = i
                max = posteriorProbabilities[i]
        if maxPos == int(page[0]):
            numCorrect += 1
        else:
            numIncorrect += 1
        if abs(maxPos - int(page[0])) <= 1:
            numBufferCorrect += 1
        confusionMatrix[maxPos][int(page[0])] += 1

    bestCorrect = numCorrect / numTestCases
    bestBufferedCorrect = numBufferCorrect / numTestCases
    print "Percentage Correct:\t" + str(numCorrect / numTestCases * 100)
    print "Percentage Incorrect:\t" + str(numIncorrect / numTestCases * 100)


    currWordsUsed = currWords[:]


    cmStorage = open("confusionMatrixWordsRemoved.txt", "a")

    cmStorage.write("Page size: " + str(pageSize) + " paragraphs\n")
    cmStorage.write("Num Training Lines: " + str(numTrainingCases) + "\n")
    cmStorage.write("Num Testing Cases:\t" + str(numTestCases) + "\n")
    cmStorage.write("Num Validation Cases:\t" + str(numValidationCases) + "\n")
    cmStorage.write("Words Used for Classifying:\t" + str(len(currWords)) + "\n")
    cmStorage.write("numNeccesary:\t" + str(numNeccesary)+"\n")
    cmStorage.write("upperThreshold:\t" + str(upperThreshold) + "\n")
    cmStorage.write("lowerThreshold:\t" + str(lowerThreshold) + "\n")
    savedWords = ""
    for i in range(len(currWords)):
        savedWords += "\t" + currWords[i]
    cmStorage.write("Words:\t" + savedWords + "\n")
    cmStorage.write("Percentage Validation Correct:\t" + str(validationBest) + "\n")
    cmStorage.write("Percentage Correct:\t" + str(numCorrect/numTestCases*100) + "\n")
    cmStorage.write("Percentage Buffered:\t" + str(bestBufferedCorrect* 100) + "\n")
    cmStorage.write("Percentage Incorrect:\t" + str(numIncorrect / numTestCases * 100) + "\n\n")
    topLine = " "
    barrierLine = "-"
    for i in range(numBooks):
        topLine += "\t|\t" + str(i+1)
        barrierLine += "---|----"
    topLine += "\t|"
    barrierLine += "---|"
    print topLine
    print barrierLine
    cmStorage.write(topLine + "\n")
    cmStorage.write(barrierLine + "\n")

    for i in range(numBooks):
        topLine = ""
        topLine += str(i+1) + " "
        for j in range(numBooks):
            topLine += "\t|\t" + str(confusionMatrix[i][j])
        print topLine + "\t|"
        cmStorage.write(topLine + "\n")
    cmStorage.write("\n\n")
    cmStorage.close()
