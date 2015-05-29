import parse
import rnn
import wordUtil
import time

# Training input files
TRAIN_FILENAME = "data/training_v5_noTag_byChoice_noWordsWithMarks_3584_1737297.txt"
TEST_FILENAME = None #"test.txt"
PROBLEM_FILENAME = "data/test_v4.txt"

# Neural Network Model saving and loading file name
#SAVE_MODEL_FILENAME = "models/rnn.model"
LOAD_MODEL_FILENAME = None #"models/RNN_CO18.0074_HL128-1_EP1_LR0.004_BS1024.model" #<- Change this if you want to train from an existing model

# Result output csv file
#OUTPUT_CSV_FILENAME = "output/result.csv"

# Nerual Network Parameters
HIDDEN_LAYER = [128]  # 1 hidden layer
BPTT_ORDER = 4
LEARNING_RATE = 0.004
EPOCH_NUM = 1  # number of epochs to run before saving the model
BATCH_SIZE = 1024


currentEpoch = 0

print 'Parsing training data...'
t0 = time.time()

trainWordIndicesOrigin = parse.parseData(TRAIN_FILENAME)

t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

LOAD_FILENAME = "trainImportant.txt"
with open(LOAD_FILENAME) as file:
    line = file.readline()
    rowList = line.rstrip().split(" ")
    trainWordIndices = [trainWordIndicesOrigin[index] for index in rowList]


# NEURON_NUM_LIST = [ HIDDEN_LAYER + [ wordUtil.WORD_VECTOR_SIZE ] ] + HIDDEN_LAYER + [ wordUtil.TOTAL_WORDS ]
NEURON_NUM_LIST = [ HIDDEN_LAYER + [ wordUtil.WORD_VECTOR_SIZE ] ] + HIDDEN_LAYER + [ [ wordUtil.WORD_CLASS_SIZE, wordUtil.WORD_CLASS_NUM ] ]

print 'Generating utils for class-based output layer...'
t0 = time.time()
wordUtil.genWordClassUtils(trainWordIndices)
#wordUtil.loadWordClassUtilsModel("models/wordClass.dat")
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Training...'
aRNN = rnn.rnn( NEURON_NUM_LIST, BPTT_ORDER, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE, LOAD_MODEL_FILENAME )

while True:
    t2 = time.time()
    aRNN.train(trainWordIndices)
    t3 = time.time()
    print '...costs ', t3 - t2, ' seconds'

    # print 'Error rate: ', aDNN.errorRate

    currentEpoch += EPOCH_NUM

    # Saving the Neural Network Model
    modelInfo = "_CO" + str(aRNN.cost)[0:7] \
        + "_HL" + str(HIDDEN_LAYER[0]) + "-" + str(len(HIDDEN_LAYER)) \
        + "_EP" + str(currentEpoch) \
        + "_LR" + str(LEARNING_RATE) \
        + "_BS" + str(BATCH_SIZE)
    SAVE_MODEL_FILENAME = "models/RNN" + modelInfo + ".model"
    aRNN.saveModel(SAVE_MODEL_FILENAME)
    
    print 'Parsing testing data...'
    t0 = time.time()
    testWordIndices = parse.parseData(PROBLEM_FILENAME)
    t1 = time.time()
    print '...costs ', t1 - t0, ' seconds'

    print 'Testing...'
    t4 = time.time()
    predictIndices = aRNN.test(testWordIndices)
    t5 = time.time()
    print '...costs', t5 - t4, ' seconds'

    print 'Writing to csv file...'
    OUTPUT_CSV_FILENAME = "output/TEST" + modelInfo + ".csv"
    parse.outputCsvFileFromAnswerNumbers(predictIndices, OUTPUT_CSV_FILENAME)

# print 'Parsing test data...'
# t0 = time.time()

# testWordVectors, testWordIndices = parse.parseData(PROBLEM_FILENAME)

# t1 = time.time()
# print '...costs ', t1 - t0, ' seconds'

# print 'Parsing problems and answers...'
# t0 = time.time()

# problems, answers = parse.parseProblemsAndAnswers(PROBLEM_FILENAME)

# t1 = time.time()
# print '...costs ', t1 - t0, ' seconds'

# print 'Dotproduct calculating...'
# t0 = time.time()

# guessAnswer = []
# for i in xrange(len(answers)):
#     degSum = []
#     for j in xrange(len(answers[i])):
#         oneDegSum = 0
#         for k in xrange(len(problems[i])):
#             oneDegSum += parse.dotproduct(answers[i][j], problems[i][k])
#         degSum.append(oneDegSum)
#     guessAnswer.append(degSum.index(max(degSum)))

# t1 = time.time()
# print '...costs ', t1 - t0, ' seconds'

# print 'Writing output file...'
# t0 = time.time()

# parse.outputCsvFileFromAnswerNumbers(guessAnswer, OUTPUT_CSV_FILENAME)

# t1 = time.time()
# print '...costs ', t1 - t0, ' seconds'