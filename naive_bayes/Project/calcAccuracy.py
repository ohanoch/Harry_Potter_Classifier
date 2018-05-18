import random
import math

numTests = 13
numBooks = 7
bookFile = open("matrices" + ".txt", "r")
paragraphSize = []
confusionMatrices = []
totalTrained = []
totalTested =[]

error = []
accuracy = []
recall = []
precision = []

def addToBook(arr, book):
    temp = "START\n"
    for i in range(numBooks):
        temp += str(i) + "\t" + str(arr[i]) + "\n"
    temp += "END\n"
    book.write(temp)

def getRecall(paragraphIndex):
    currRecall = []
    matrix = confusionMatrices[paragraphIndex]
    print paragraphIndex
    for i in range(numBooks):
        FP = 0.0
        TP = 0.0
        for j in range(numBooks):
            if i != j:
                FP += matrix[i][j]
            else:
                TP += matrix[j][i]
        if TP == 0 and FP == 0:
            currRecall.append(0)
        else:
            currRecall.append(TP /(TP + FP))

    return currRecall

def getPrecision(paragraphIndex):
    currPrecision = []
    matrix = confusionMatrices[paragraphIndex]
    total = totalTested[paragraphIndex]
    for i in range(numBooks):
        FN = 0.0
        TP = 0.0
        for j in range(numBooks):
            if i != j:
                FN += matrix[j][i]
            else:
                TP += matrix[j][i]
        currPrecision.append(TP /(TP + FN))

    return currPrecision


def getError(paragraphIndex):
    currError = []
    matrix = confusionMatrices[paragraphIndex]
    total = totalTested[paragraphIndex]
    for i in range(numBooks):
        FP = 0.0
        FN = 0.0
        for j in range(numBooks):
            if i != j:
                FP += matrix[i][j]
                FN += matrix[j][i]
        currError.append((FP + FN) / total)

    return currError

def getAccuracy(currError):
    currAccuracy = []
    for i in range(numBooks):
        currAccuracy.append(1-currError[i])

    return currAccuracy

for i in range(numTests):
    currMatrix = []
    for j in range(numBooks):
        currLine = bookFile.readline()
        currRow = currLine.split("\t|\t")
        currRow = map(int, currRow)
        currMatrix.append(currRow)

    confusionMatrices.append(currMatrix)

sum = [0]*7
storage = [0]*7

for i in [2]:
    matrix = confusionMatrices[i]
    sum = [0, 0, 0, 0, 0, 0, 0]
    total = 0.0
    for j in range(numBooks):
        for k in range(numBooks):
            if not (j == k):
                total += matrix[j][k]
                sum[j] += matrix[j][k]

    for j in range(numBooks):
        storage[j] += sum[j]/ (total)


toWrite = open("probs.txt", "w")

addToBook(storage, toWrite)
