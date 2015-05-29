import parse
import wordUtil
from rnnlm import RNNLM
import time

# Word vectors file
WORD_VECTORS_FILENAME = "data/smallTest/vec.txt"

# Training / Testing input files
TRAIN_FILENAME = "data/smallTest/training.txt"
TEST_FILENAME = "data/smallTest/testing.txt"

# Neural Network Model saving and loading file name
SAVE_MODEL_FILENAME = "models/rnn.model"
# LOAD_MODEL_FILENAME = None #"models/rnn.model" <- Change this if you want to train from an existing model
LOAD_MODEL_FILENAME = "models/rnn.model"

# Result output csv file
OUTPUT_CSV_FILENAME = "output/result.csv"

# Test data choice number
NUM_OF_CHOICES = 5

# Nerual Network Parameters
HIDDEN_LAYER = [128]  # 1 hidden layer
BPTT_ORDER = 4
LEARNING_RATE = 0.25
EPOCH_NUM = 200  # number of epochs to run before saving the model
TRAIN_BATCH_SIZE = 3
TEST_BATCH_SIZE = 15

print 'Parsing word vectors...'
t0 = time.time()
wordUtil.parseWordVectors(WORD_VECTORS_FILENAME)
t1 = time.time()
print '...costs', t1 - t0, 'seconds'

# print 'Parsing training data...'
# t0 = time.time()
# trainWordIndices = parse.parseAndClusterTrainData(TRAIN_FILENAME, TRAIN_BATCH_SIZE)
# t1 = time.time()
# print '...costs', t1 - t0, 'seconds'

print 'Parsing testing data...'
t0 = time.time()
testWordIndices = parse.parseData(TEST_FILENAME)
t1 = time.time()
print '...costs', t1 - t0, 'seconds'

NEURON_NUM_LIST = [ HIDDEN_LAYER + [ wordUtil.WORD_VECTOR_SIZE ] ] + HIDDEN_LAYER + [ [wordUtil.TOTAL_WORDS] ]
aRNNLM = RNNLM(NEURON_NUM_LIST, SAVE_MODEL_FILENAME, LOAD_MODEL_FILENAME)

# print 'Training...'
# t0 = time.time()
# aRNNLM.train(EPOCH_NUM, TRAIN_BATCH_SIZE, BPTT_ORDER, LEARNING_RATE, trainWordIndices)
# t1 = time.time()
# print '...costs', t1 - t0, 'seconds'

print 'Testing...'
t0 = time.time()
answers = aRNNLM.test(TEST_BATCH_SIZE, NUM_OF_CHOICES, testWordIndices)
t1 = time.time()
print '...costs', t1 - t0, 'seconds'

# print 'Writing to csv file...'
# parse.outputCsvFileFromAnswerNumbers(answers, OUTPUT_CSV_FILENAME)
