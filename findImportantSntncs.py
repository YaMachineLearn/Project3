# import parse
import re
from theano import shared
import rnn
import wordUtil
import time
import numpy as np

# Training input files
# TRAIN_FEATURE_FILENAME = "vec.txt"
# TEST_FILENAME = "test.txt"

LOAD_MODEL_FILENAME = None

# Nerual Network Parameters
HIDDEN_LAYER = [8]  # 1 hidden layer
BPTT_ORDER = 3
LEARNING_RATE = 1.0
EPOCH_NUM = 50  # number of epochs to run before saving the model
BATCH_SIZE = 2

print 'Training...'
# trainFeats = [
#     [ i am a student ],
#     [ he is not a student however ]
# ]
trainLabels = [
    [1,2,3,4],
    [5,6,7,3,4],
    [2,3,4],
    [5,6]
]
testLabels = [
    [1,2,3,4],
    [1,1,3,4],
    [5,6,7,3,6],
    [5,6,7,3,4],
    [2,3,4],
    [5,3,4],
    [5,6],
    [5,7]
]

TOTAL_WORDS = 8
# NEURON_NUM_LIST = [ HIDDEN_LAYER + [ wordUtil.WORD_VECTOR_SIZE ] ] + HIDDEN_LAYER + [ wordUtil.TOTAL_WORDS ]
NEURON_NUM_LIST = [ HIDDEN_LAYER + [ wordUtil.WORD_VECTOR_SIZE ] ] + HIDDEN_LAYER + [ [ wordUtil.WORD_CLASS_SIZE, wordUtil.WORD_CLASS_NUM ] ]

wordUtil.genWordClassUtils(trainLabels)

numOfChoices = 2
    
def parseChoices(RAW_TEST_FILE):
    pattern1 = "(.+\[)([^\]]+)"  #get anything inside [ ]
    #pattern2 = "(\w)[(^\w)]?[(\w)]?"   #split with any symbols
    with open(RAW_TEST_FILE) as testFile:
        i = 1
        choicesWordList = list()
        for line in testFile:   
            tempSet = set()         
            m1 = re.match(pattern1, line)
            choice = m1.group(2)

            choice = choice.lower()
            if ("-" in choice) or ("'" in choice) or ("," in choice):
                pass
            else:
                tempSet.add(choice)
            if i % numOfChoices == 0:
                choicesWordList.append(tempSet)
            i += 1
    return choicesWordList

def findImportantSntncs(trainLabels, testLabels, choicesWordList):
    returnSntncIndices = set()
    for i in xrange(len(choicesWordList)):
        newTrainLabels = []
        for sntnc in trainLabels:
            for word in sntnc:
                if word in choicesWordList[i]:
                    newTrainLabels.append(sntnc)
                    break
        testContain = set()
        for j in xrange(numOfChoices):
            testContain = testContain | set(testLabels[i * numOfChoices + j])
        crossCount = np.zeros( len(trainLabels), dtype='int32' )
        for j in xrange(len(trainLabels)):
            crossCount[j] = len(testContain.intersection(trainLabels[j]))
        maxIndices = np.argpartition(crossCount, -2)[-2:]
        for j in xrange(len(maxIndices)):
            returnSntncIndices.add(maxIndices[j])
    return [trainLabels[index] for index in returnSntncIndices]

RAW_TEST_FILE = "testing_data.txt"
choicesWordList = parseChoices(RAW_TEST_FILE)
choicesWordList = [set([2, 1]), set([6, 4]), set([2, 5]), set([6, 7])]
trainImportantLabels = findImportantSntncs(trainLabels, testLabels, choicesWordList)
print trainImportantLabels

LOAD_FILENAME = "trainImportant.txt"
with open(LOAD_FILENAME) as file:
    line = file.readline()
    rowList = line.rstrip().split(" ")
    newTrainWordIndices = [trainWordIndices[index] for index in rowList]

aRNN = rnn.rnn( NEURON_NUM_LIST, BPTT_ORDER, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE, LOAD_MODEL_FILENAME )
aRNN.train(trainImportantLabels)


print 'Testing...'
predictIndices = aRNN.test(testLabels)

# print 'Saving model...'
# def outputCsvFileFromAnswerNumbers(guessAnswer, OUTPUT_FILE):
#     with open(OUTPUT_FILE, 'w') as outputFile:
#         outputFile.write('Id,Answer\n')
#         for i in xrange(len(guessAnswer)):
#             outputFile.write(str(i + 1) + ',' + chr(97 + guessAnswer[i]) + '\n' )
# OUTPUT_CSV_FILENAME = "output/TEST.csv"
# outputCsvFileFromAnswerNumbers(predictIndices, OUTPUT_CSV_FILENAME)
