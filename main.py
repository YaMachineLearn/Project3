import parse
# import dnn
# import labelUtil
import time

# Training input files
TRAIN_FILENAME = "training.txt"
TEST_FILENAME = None #"test.txt"
PROBLEM_FILENAME = "test.txt"

# Neural Network Model saving and loading file name
SAVE_MODEL_FILENAME = None #"models/dnn.model"
LOAD_MODEL_FILENAME = None #"models/dnn.model" <- Change this if you want to train from an existing model

# Result output csv file
OUTPUT_CSV_FILENAME = "output/result.csv"

# Nerual Network Parameters
HIDDEN_LAYER = [128]  # 1 hidden layer
BPTT_ORDER = 3
LEARNING_RATE = 0.05
EPOCH_NUM = 10  # number of epochs to run before saving the model
BATCH_SIZE = 256

currentEpoch = 1

print 'Parsing training data to word vectors...'
t0 = time.time()

trainWordVectors = parse.parseTrainDataToWordVectors(TRAIN_FILENAME)

t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Parsing problems to word vectors...'
t0 = time.time()

problem, answers = parse.parseProblemToWordVectors(PROBLEM_FILENAME)

t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Dotproduct calculating...'
t0 = time.time()

guessAnswer = []
for i in xrange(len(answers)):
    degSum = []
    for j in xrange(len(answers[i])):
        oneDegSum = 0
        for k in xrange(len(problem[i])):
            oneDegSum += parse.dotproduct(answers[i][j], problem[i][k])
        degSum.append(oneDegSum)
    guessAnswer.append(degSum.index(max(degSum)))

t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Writing output file...'
t0 = time.time()

parse.outputCsvFileFromAnswerNumbers(guessAnswer, OUTPUT_CSV_FILENAME)

t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

"""
NEURON_NUM_LIST = [ HIDDEN_LAYER + [ len(trainFeats[0]) ] ] + HIDDEN_LAYER + [ labelUtil.LABEL_NUM ]

print 'Training...'
aDNN = dnn.dnn( NEURON_NUM_LIST, BPTT_ORDER, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE, LOAD_MODEL_FILENAME )

while True:
    t2 = time.time()
    aDNN.train(trainFeats, trainLabels)
    t3 = time.time()
    print '...costs ', t3 - t2, ' seconds'

    print 'Error rate: ', aDNN.errorRate

    currentEpoch += EPOCH_NUM
    
    # Saving the Neural Network Model
    modelInfo = "_ER" + str(aDNN.errorRate)[2:5] \
        + "_CO" + str(aDNN.cost)[0:7] \
        + "_HL" + str(HIDDEN_LAYER[0]) + "-" + str(len(HIDDEN_LAYER)) \
        + "_EP" + str(currentEpoch) \
        + "_LR" + str(LEARNING_RATE) \
        + "_BS" + str(BATCH_SIZE)
    SAVE_MODEL_FILENAME = "models/DNN" + modelInfo + ".model"
    aDNN.saveModel(SAVE_MODEL_FILENAME)

    print 'Testing...'
    t4 = time.time()
    testLabels = aDNN.test(testFeats)
    t5 = time.time()
    print '...costs', t5 - t4, ' seconds'

    print 'Writing to csv file...'
    OUTPUT_CSV_FILE_NAME = "output/TEST" + modelInfo + ".csv"
    parse.outputTestLabelAsCsv(testFrameNames, testLabels, OUTPUT_CSV_FILE_NAME)
"""