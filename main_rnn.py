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
    [5,6,7,3,4]
]

TOTAL_WORDS = 8
NEURON_NUM_LIST = [ HIDDEN_LAYER + [ wordUtil.WORD_VECTOR_SIZE ] ] + HIDDEN_LAYER + [ wordUtil.TOTAL_WORDS ]

aRNN = rnn.rnn( NEURON_NUM_LIST, BPTT_ORDER, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE, LOAD_MODEL_FILENAME )
aRNN.train(trainLabels)

testLabels = [
    [1,2,3,4],
    [1,1,3,4],
    [5,6,7,3,6],
    [5,6,7,3,4]
]
print 'Testing...'
aRNN.test(testLabels)
