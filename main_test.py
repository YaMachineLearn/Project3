# import parse
import dnn
import wordUtil
import time

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
trainFeats = [
    [ [0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0] ],
    [ [0,0,0,0,0,0,0,1], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,1,0,0,0,0,0] ]
]
trainLabels = [
    [1,2,3,4],
    [5,6,7,3,4]
]
LABEL_NUM = 8

NEURON_NUM_LIST = [ HIDDEN_LAYER + [ len(trainFeats[0][0]) ] ] + HIDDEN_LAYER + [ LABEL_NUM ]

aDNN = dnn.dnn( NEURON_NUM_LIST, BPTT_ORDER, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE, LOAD_MODEL_FILENAME )
aDNN.train(trainLabels)

testFeats = [
    [ [0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0] ],
    [ [0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0] ],
    [ [0,0,0,0,0,0,0,1], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,1,0,0,0,0,0] ],
    [ [0,0,0,0,0,0,0,1], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,1,0,0,0,0,0] ]
]
testLabels = [
    [0,1,2,3],
    [0,0,2,3],
    [4,5,6,2,5],
    [4,5,6,2,3]
]
aDNN.test(testFeats, testLabels)

"""
currentEpoch = 1
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