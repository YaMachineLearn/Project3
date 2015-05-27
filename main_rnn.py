# import parse
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
EPOCH_NUM = 100  # number of epochs to run before saving the model
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

TOTAL_WORDS = 8
# NEURON_NUM_LIST = [ HIDDEN_LAYER + [ wordUtil.WORD_VECTOR_SIZE ] ] + HIDDEN_LAYER + [ wordUtil.TOTAL_WORDS ]
NEURON_NUM_LIST = [ HIDDEN_LAYER + [ wordUtil.WORD_VECTOR_SIZE ] ] + HIDDEN_LAYER + [ [ wordUtil.WORD_CLASS_SIZE, wordUtil.WORD_CLASS_NUM ] ]

wordUtil.genWordClassUtils(trainLabels)
aRNN = rnn.rnn( NEURON_NUM_LIST, BPTT_ORDER, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE, LOAD_MODEL_FILENAME )
aRNN.train(trainLabels)

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
print 'Testing...'
predictIndices = aRNN.test(testLabels)

print 'Saving model...'
def outputCsvFileFromAnswerNumbers(guessAnswer, OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w') as outputFile:
        outputFile.write('Id,Answer\n')
        for i in xrange(len(guessAnswer)):
            outputFile.write(str(i + 1) + ',' + chr(97 + guessAnswer[i]) + '\n' )
OUTPUT_CSV_FILENAME = "output/TEST.csv"
outputCsvFileFromAnswerNumbers(predictIndices, OUTPUT_CSV_FILENAME)