import time
import labelUtil
import random
import theano
import numpy as np
import sys
import theano.tensor as T
from theano import shared
from theano import function

class dnn:
    def __init__(self, neuronNumList, bpttOrder, learningRate, epochNum, batchSize, LOAD_MODEL_FILENAME=None):
        self.neuronNumList = neuronNumList    #ex: [69, 128, 128, 128, 48]
        self.bpttOrder = bpttOrder
        self.learningRate = learningRate
        self.epochNum = epochNum
        self.batchSize = batchSize
        #better: check if input parameters are in correct formats
        
        self.weightMatrices = []
        if LOAD_MODEL_FILENAME is None:
            self.setRandomModel()
        else:
            self.loadModel(LOAD_MODEL_FILENAME)
        #ex: weightMatrices == [ [ [,,],[,,],...,[,,] ], [ [,,],[,,],...,[,,] ], ... ]
        
    def train(self, trainFeats, trainLabels):
        indices = T.ivector()
        trainFeatsArray = shared(np.transpose(np.asarray(trainFeats, dtype=theano.config.floatX)))
        inputVector = trainFeatsArray[:, indices]
        # trainLabelsArray = shared(np.transpose(labelUtil.labelToArray(trainLabels)))
        outputRef = trainLabels[:, indices]
        self.initLastHiddenOut()
        lineIn_h = self.lastHiddenOut
        lineIn_i = inputVector ############# To be modified ##############
        for i in range( self.bpttOrder ):
            weightMatrix_h = self.weightMatrices[i][0]
            weightMatrix_i = self.weightMatrices[i][1]
            lineOutput = T.dot(weightMatrix_h, lineIn_h) + T.dot(weightMatrix_i, lineIn_i)
            lineIn_h = 1. / (1. + T.exp(-lineOutput)) # the output of the current layer is the input of the next layer
        weightMatrix_o = self.weightMatrices[self.bpttOrder][0]
        lineOutput = T.dot(weightMatrix_o, lineIn_h)
        outputVector = ( T.nnet.softmax(lineOutput.T) ).T # .T means transpose
        cost = T.sum( - T.log(outputVector[:, outputRef]) ) / self.batchSize ########## To be checked ###########
        params = self.weightMatrices
        gparams = [T.grad(cost, param) for param in params]
        train_model = function(inputs=[indices], outputs=[outputVector, cost], updates=self.update(params, gparams))

        #start training
        numOfBatches = len(trainFeats) / self.batchSize
        shuffledIndex = range(len(trainFeats))
        for epoch in xrange(self.epochNum):
            # shuffle feats and labels
            random.shuffle(shuffledIndex)

            count = 0
            sumCost = 0.
            for i in xrange(numOfBatches): #feats and labels are shuffled, so don't need random index here
                progress = float(count + (numOfBatches * epoch)) / float(numOfBatches * self.epochNum) * 100.
                sys.stdout.write('Epoch %d, Progress: %f%%    \r' % (epoch, progress))
                sys.stdout.flush()
                self.out, self.cost = train_model(shuffledIndex[i*self.batchSize:(i+1)*self.batchSize])
                sumCost = sumCost + self.cost
                count = count + 1
            self.cost = sumCost / float(numOfBatches)
            print 'Cost: ', sumCost / float(numOfBatches)

        self.calculateError(trainFeats, trainLabels)

    def test(self, testFeats):
        test_model = self.getForwardFunction(testFeats, len(testFeats), self.weightMatrices)
        testLabels = []
        outputArray = test_model(0)
        outputMaxIndex = T.argmax(outputArray, 0).eval()
        for i in xrange(len(outputMaxIndex)):
            testLabels.append(labelUtil.LABEL_LIST[outputMaxIndex[i]])
        return testLabels

    def forward(self):
        pass

    def getForwardFunction(self, testFeats, batchSize, weightMatrices):
        index = T.iscalar()
        testFeatsArray = shared(np.transpose(np.asarray(testFeats, dtype=theano.config.floatX)))
        inputVectorArray = testFeatsArray[:, index * batchSize:(index + 1) * batchSize]
        lineIn = inputVectorArray
        for i in range( len(weightMatrices) ):
            weightMatrix = weightMatrices[i]
            lineOutput = T.dot(weightMatrix, lineIn)
            lineIn = 1. / (1. + T.exp(-lineOutput)) # the output of the current layer is the input of the next layer
        outputVectorArray = lineIn
        test_model = function(inputs=[index], outputs=outputVectorArray)
        return test_model

    def backProp(self):
        pass

    def update(self, params, gparams):
        updates = [(param, param - self.learningRate * gparam) for param, gparam in zip(params, gparams)]
        return updates

    def calculateError(self, trainFeats, trainLabels):
        batchNum = 7   #1124823 = 3*7*29*1847
        calcErrorSize = len(trainFeats) / batchNum
        forwardFunction = self.getForwardFunction(trainFeats, calcErrorSize, self.weightMatrices)
        self.errorNum = 0
        for i in xrange(batchNum):
            forwardOutput = forwardFunction(i)
            self.errorNum += np.sum(T.argmax(forwardOutput, 0).eval() != labelUtil.labelsToIndices(trainLabels[i*calcErrorSize:(i+1)*calcErrorSize]))
        self.errorRate = self.errorNum / float(calcErrorSize * batchNum)

    def initLastHiddenOut(self):
        lastHiddenOut = shared( np.zeros( (self.neuronNumList[0][0], 1), dtype=theano.config.floatX ) )

    ### Model generate, save and load ###
    def setRandomModel(self):
        w_h = np.asarray( np.random.normal(
            loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[i]),
            size=(self.neuronNumList[1], self.neuronNumList[0][0])), dtype=theano.config.floatX)
        w_i = np.asarray( np.random.normal(
            loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[i]),
            size=(self.neuronNumList[1], self.neuronNumList[0][1])), dtype=theano.config.floatX)
        w_o = np.asarray( np.random.normal(
            loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[i]),
            size=(self.neuronNumList[2], self.neuronNumList[1])), dtype=theano.config.floatX)
        for i in range( 5 ):    #ex: range(5-1) => 0, 1, 2, 3
            self.weightMatrices.append( [ shared(w_h) ] + [ shared(w_i) ] )
        self.weightMatrices.append( shared(w_o) )

    def saveModel(self, SAVE_MODEL_FILENAME):
        with open(SAVE_MODEL_FILENAME, 'w') as outputModelFile:
            for i in xrange( len(self.weightMatrices) * 2 ):
                # Saving weight matrices
                if i % 2 == 0:
                    weightMatrix = np.asarray(self.weightMatrices[i / 2].get_value(borrow=True, return_internal_type=True))
                    weightMatrixDim = weightMatrix.shape  # Shape (matrix height, matrix width)
                    for row in xrange( weightMatrixDim[0] ):
                        for col in xrange( weightMatrixDim[1] ):
                            outputModelFile.write(str(weightMatrix[row][col]) + ' ')
                        outputModelFile.write('\n')
                    outputModelFile.write('\n')
                # Saving bias arrays
                else:
                    biasVector = np.asarray(self.biasArrays[(i - 1) / 2].get_value(borrow=True, return_internal_type=True))
                    biasVectorDim = biasVector.shape  # Shape (vector height, vector width)
                    for row in xrange( biasVectorDim[0] ):
                        outputModelFile.write(str(biasVector[row][0]) + ' ')
                    outputModelFile.write('\n\n')

    def loadModel(self, LOAD_MODEL_FILENAME):
        print 'Loading Neural Network Model...'
        t0 = time.time()
        with open(LOAD_MODEL_FILENAME) as modelFile:
            i = 0
            weightMatrix = []
            biasVector = []
            for line in modelFile:
                if i < (len(self.neuronNumList) - 1) * 2:
                    if line == '\n':
                        if i % 2 == 0:
                            self.weightMatrices.append(shared(np.asarray(weightMatrix)))
                            weightMatrix = []
                        else:
                            self.biasArrays.append(shared(np.asarray(biasVector)))
                            biasVector = []
                        i = i + 1
                    if line.rstrip():
                        rowList = line.rstrip().split(" ")
                        if i % 2 == 0:
                            weightMatrix.append([float(ele) for ele in rowList])
                        else:
                            for ele in rowList:
                                biasVector.append([float(ele)])
        t1 = time.time()
        print '...costs ', t1 - t0, ' seconds'
