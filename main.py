import parse
# import dnn
# import labelUtil
import time

# Training input files
TRAIN_FEATURE_FILENAME = "vec.txt"
TEST_FILENAME = "test.txt"

# Neural Network Model saving and loading file name
SAVE_MODEL_FILENAME = None #"models/dnn.model"
LOAD_MODEL_FILENAME = None #"models/dnn.model" <- Change this if you want to train from an existing model

# Result output csv file
OUTPUT_CSV_FILE_NAME = None #"output/result.csv"

# Nerual Network Parameters
HIDDEN_LAYER = [128]  # 1 hidden layer
BPTT_ORDER = 3
LEARNING_RATE = 0.05
EPOCH_NUM = 10  # number of epochs to run before saving the model
BATCH_SIZE = 256

currentEpoch = 1

print 'Parsing...'
t0 = time.time()
trainFeats, trainLabels = parse.parseTrainFeatures(TRAIN_FEATURE_FILENAME)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Problems preprocessing...'
t0 = time.time()
problem, answers = parse.parseTestFeatures(TEST_FILENAME, trainFeats, trainLabels)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'
"""
print parse.length(problem[31][3])
v1 = answers[31][2]
v2 = problem[31][3]
print parse.dotproduct(v1, v2) / (parse.length(v1) * parse.length(v2))
"""

print 'Degree calculating...'
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

OUTPUT_FILE = "output.csv"
 
""" writing output file """
print "writing output file..."
with open(OUTPUT_FILE, 'w') as outputFile:
    outputFile.write('Id,Answer\n')
    for i in xrange(1040):
        outputFile.write( str(i+1) + ',' + chr(97+guessAnswer[i]) + '\n' )

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