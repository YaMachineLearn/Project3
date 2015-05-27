import time
import wordUtil
import random
import theano
import numpy as np
import sys
import theano.tensor as T
from theano import shared
from theano import function

class rnn:
    def __init__(self, neuronNumList, bpttOrder, learningRate, epochNum, batchSize, LOAD_MODEL_FILENAME=None):
        self.neuronNumList = neuronNumList    #ex: [69, 128, 128, 128, 48]
        self.bpttOrder = bpttOrder
        self.learningRate = learningRate
        self.epochNum = epochNum
        self.batchSize = batchSize
        #better: check if input parameters are in correct formats

        self.lastHiddenOut = self.initLastHiddenOut()
        
        self.weightMatrices = []
        if LOAD_MODEL_FILENAME is None:
            self.setRandomModel()
        else:
            self.loadModel(LOAD_MODEL_FILENAME)
        #ex: weightMatrices == [ [ [,,],[,,],...,[,,] ], [ [,,],[,,],...,[,,] ], ... ]
        
    def train(self, trainLabels):
        # better: trainFeatsArray should be moved outside?
        # trainFeatsArray is zero-padded at the beginning of each sentence
        trainSntncLengths = []
        maxLength = 1
        for sentence in trainLabels:
            trainSntncLengths.append( len(sentence) )
            if ( len(sentence) > maxLength ):
                maxLength = len(sentence)
        for i in xrange(len(trainLabels)):
            # trainFeats[i].extend( [ [0] * self.neuronNumList[0][1] for j in xrange(maxLength - trainSntncLengths[i]) ] )
            trainLabels[i].extend( [0] * (maxLength - trainSntncLengths[i]) )

        # trainFeatsArray = shared( np.concatenate( (
        #     np.zeros((len(trainFeats), self.bpttOrder - 1, self.neuronNumList[0][1]), dtype=theano.config.floatX),
        #     np.asarray(trainFeats, dtype=theano.config.floatX) ), axis=1 ) )
        # trainFeatsArray = shared( np.asarray(trainFeats, dtype=theano.config.floatX) )
        trainSntncLengthsVec = shared( np.asarray(trainSntncLengths, dtype='int32') )

        # sntncIndex = T.iscalar('sntncIndex')
        sntncIndices = T.ivector('sntncIndices')
        wordIndex = T.iscalar('wordIndex')
        trainLabelsArray = shared(np.asarray(trainLabels, dtype='int32'))

        # add <START> at the start of each sentence
        trainFeatsArray = T.concatenate(  [ shared( np.zeros( (len(trainLabels), 1), dtype='int32' ) ), trainLabelsArray[:, 0 : maxLength -1] ], axis=1 )
        # outputRef = trainLabelsArray[sntncIndex, wordIndex]
        # outputRef = trainLabelsArray[sntncIndices, wordIndex]
        outputRef = wordUtil.WORD_CLASS_LABELS[trainLabelsArray[sntncIndices, wordIndex]]
        outputClassRef = wordUtil.WORD_CLASSES[trainLabelsArray[sntncIndices, wordIndex]]
        # print wordUtil.WORD_VECTORS[ trainLabelsArray[0:2, 0] ].dimshuffle(0, 'x', 1).eval()
        # print self.weightMatrices[2][ wordUtil.WORD_CLASSES[0:2] ].eval()

        # def recurrence(x_t, h_tm1): 
        def recurrence(x_t, h_tm1): 
            # h_t = T.nnet.sigmoid( T.dot(self.weightMatrices[0], h_tm1) + T.dot(self.weightMatrices[1], x_t) )
            h_t = T.nnet.sigmoid( T.dot(h_tm1, self.weightMatrices[0]) + T.dot(x_t, self.weightMatrices[1]) )
            return h_t

        startIndex = T.maximum( 0, wordIndex - self.bpttOrder + 1 )
        # x = wordUtil.WORD_VECTORS[trainFeatsArray[sntncIndices, startIndex : wordIndex + 1]].dimshuffle(1, 2, 0)
        x = wordUtil.WORD_VECTORS[trainFeatsArray[sntncIndices, startIndex : wordIndex + 1]].dimshuffle(1, 0, 2)
        h, _ = theano.scan(fn=recurrence, sequences=x, outputs_info=self.lastHiddenOut, n_steps=x.shape[0])
        # cost = - T.sum( T.switch( T.lt(wordIndex, trainSntncLengthsVec[sntncIndices]), T.log(y[-1][outputRef, range(self.batchSize)]), 0.0 ) )
        # y = T.nnet.softmax( T.dot(h[-1], self.weightMatrices[2]) )
        y_o = T.nnet.softmax( T.batched_dot(h[-1].dimshuffle(0, 'x', 1), self.weightMatrices[2][ wordUtil.WORD_CLASSES[outputClassRef] ]).reshape([self.batchSize, wordUtil.WORD_CLASS_SIZE]) )
        y_c = T.nnet.softmax( T.dot(h[-1], self.weightMatrices[3]) )
        # cost = - T.sum( T.switch( T.lt(wordIndex, trainSntncLengthsVec[sntncIndices]), T.log(y[range(self.batchSize), outputRef]), 0.0 ) )
        cost = - T.sum( T.switch( T.lt(wordIndex, trainSntncLengthsVec[sntncIndices]), T.log(y_o[range(self.batchSize), outputRef]) + T.log(y_c[range(self.batchSize), outputClassRef]), 0.0 ) )
        # train_model = function(inputs=[sntncIndices, wordIndex], outputs=[y, cost], updates=self.updateScan(cost, h[0], T.gt(startIndex, 0), T.eq(wordIndex, maxLength - 1)))
        train_model = function(inputs=[sntncIndices, wordIndex], outputs=[y_o, y_c, cost], updates=self.updateScan(cost, h[0], T.gt(startIndex, 0), T.eq(wordIndex, maxLength - 1)))

        # Start training...
        numOfBatches = len(trainLabels) / self.batchSize
        shuffledIndex = range(len(trainLabels))
        for epoch in xrange(self.epochNum):
            # shuffle feats and labels
            random.shuffle(shuffledIndex)
            print '- Epoch', epoch + 1

            count = 0
            for i in xrange(numOfBatches): #feats and labels are shuffled, so don't need random index here
                sumCost = 0.
                progress = float(count + (numOfBatches * epoch)) / float(numOfBatches * self.epochNum) * 100.
                # sys.stdout.write('Epoch %d, Progress: %f%%    \r' % (epoch, progress))
                # sys.stdout.flush()
                self.initLastHiddenOut()
                for j in xrange( max([ trainSntncLengths[index] for index in shuffledIndex[i*self.batchSize : (i+1)*self.batchSize] ]) ):
                    self.wordIndex = j
                    # self.out, self.cost = ( train_models[min(j, self.bpttOrder - 1)] )(shuffledIndex[i], j)
                    # self.out, self.cost = ( train_models[min(j, self.bpttOrder - 1)] )(shuffledIndex[i*self.batchSize : (i+1)*self.batchSize], j)
                    # self.out, self.cost = train_model(shuffledIndex[i*self.batchSize : (i+1)*self.batchSize], j)
                    self.out, self.outClass, self.cost = train_model(shuffledIndex[i*self.batchSize : (i+1)*self.batchSize], j)
                    # print 'Cost: ', self.cost
                    # print 'Out: ', self.out
                    sumCost = sumCost + self.cost
                count = count + 1
                self.cost = sumCost / float(numOfBatches)
                print 'Cost: ', sumCost / float(numOfBatches), ', Progress: ', progress, ' %'

        # self.calculateError(trainFeats, trainLabels)

    def test(self, testLabels):
        numOfChoices = 5
        self.lastHiddenOut = self.initLastHiddenOut(len(testLabels))
        test_model, maxLength, testSntncLengths = self.getForwardFunction(testLabels, len(testLabels), self.weightMatrices)
        sntncProbs = np.ones(len(testLabels), dtype=theano.config.floatX)
        for i in xrange(maxLength):
            outputArray, outputClassArray = test_model(0, i)
            for j in xrange(len(testLabels)):
                if ( i < testSntncLengths[j] ):
                    # sntncProbs[j] *= outputArray[testLabels[j][i], j]
                    # sntncProbs[j] *= outputArray[j, testLabels[j][i]]
                    sntncProbs[j] *= outputArray[j, wordUtil.WORD_CLASS_LABELS[testLabels[j][i]].eval()] * outputClassArray[j, wordUtil.WORD_CLASSES[testLabels[j][i]].eval()]
            print 'sntncProbs: ', sntncProbs
        predictLabels = [ np.argmax(sntncProbs[i * numOfChoices : (i+1) * numOfChoices]) for i in xrange(len(testLabels) / numOfChoices) ]
        print predictLabels
        return predictLabels

    def forward(self):
        pass

    def getForwardFunction(self, testLabels, batchSize, weightMatrices):
        testSntncLengths = []
        maxLength = 1
        for sentence in testLabels:
            testSntncLengths.append( len(sentence) )
            if ( len(sentence) > maxLength ):
                maxLength = len(sentence)
        for i in xrange(len(testLabels)):
            # testFeats[i].extend( [ [0] * self.neuronNumList[0][1] for j in xrange(maxLength - testSntncLengths[i]) ] )
            testLabels[i].extend( [0] * (maxLength - testSntncLengths[i]) )

        # testFeatsArray = shared( np.asarray(testFeats, dtype=theano.config.floatX) )

        sntncIndex = T.iscalar()
        wordIndex = T.iscalar()
        testLabelsArray = shared(np.asarray(testLabels, dtype='int32'))
        testFeatsArray = T.concatenate(  [ shared( np.zeros( (len(testLabels), 1), dtype='int32' ) ), testLabelsArray[:, 0 : maxLength -1] ], axis=1 )
        # outputRef = testLabelsArray[sntncIndex * batchSize : (sntncIndex+1) * batchSize, wordIndex]
        outputRef = wordUtil.WORD_CLASS_LABELS[testLabelsArray[sntncIndex * batchSize : (sntncIndex+1) * batchSize, wordIndex]]
        outputClassRef = wordUtil.WORD_CLASSES[testLabelsArray[sntncIndex * batchSize : (sntncIndex+1) * batchSize, wordIndex]]
        lineIn_h = self.lastHiddenOut
        
        # lineIn_i = (wordUtil.WORD_VECTORS[testFeatsArray[sntncIndex * batchSize : (sntncIndex+1) * batchSize, wordIndex]]).T # .T means transpose
        lineIn_i = wordUtil.WORD_VECTORS[testFeatsArray[sntncIndex * batchSize : (sntncIndex+1) * batchSize, wordIndex]]
        weightMatrix_h = self.weightMatrices[0]
        weightMatrix_i = self.weightMatrices[1]
        # lineOutput_h = T.dot(weightMatrix_h, lineIn_h) + T.dot(weightMatrix_i, lineIn_i)
        lineOutput_h = T.dot(lineIn_h, weightMatrix_h) + T.dot(lineIn_i, weightMatrix_i)
        lineIn_h = T.nnet.sigmoid(lineOutput_h)

        # weightMatrix_o = self.weightMatrices[2 * self.bpttOrder]
        weightMatrix_o = self.weightMatrices[2][ wordUtil.WORD_CLASSES[outputClassRef] ]
        weightMatrix_c = self.weightMatrices[3]
        # lineOutput_o = T.dot(weightMatrix_o, lineIn_h)
        # lineOutput_o = T.dot(lineIn_h, weightMatrix_o)
        lineOutput_o = T.batched_dot(lineIn_h.dimshuffle(0, 'x', 1), weightMatrix_o).reshape([batchSize, wordUtil.WORD_CLASS_SIZE])
        lineOutput_c = T.dot(lineIn_h, weightMatrix_c)
        # outputVector = ( T.nnet.softmax(lineOutput_o.T) ).T # .T means transpose
        outputVector = T.nnet.softmax(lineOutput_o) 
        outputClassVector = T.nnet.softmax(lineOutput_c) 

        # test_model = function(inputs=[sntncIndex, wordIndex], outputs=outputVector, updates=self.updateTest(lineIn_h, T.eq(wordIndex, maxLength - 1), batchSize))
        test_model = function(inputs=[sntncIndex, wordIndex], outputs=[outputVector, outputClassVector], updates=self.updateTest(lineIn_h, T.eq(wordIndex, maxLength - 1), batchSize))
        return [test_model, maxLength, testSntncLengths]

    def backProp(self):
        pass

    def update(self, cost, index, lastHiddenOutUpdate, sntncEnd):
        totalGradW_h = T.grad(cost=cost, wrt=self.weightMatrices[0])
        totalGradW_i = T.grad(cost=cost, wrt=self.weightMatrices[1])
        for i in range( 1, self.bpttOrder ):
            totalGradW_h += T.grad(cost=cost, wrt=self.weightMatrices[2 * i], disconnected_inputs='warn')
            totalGradW_i += T.grad(cost=cost, wrt=self.weightMatrices[2 * i + 1], disconnected_inputs='warn')
        updates = []
        for i in range( self.bpttOrder ):
            updates.append( (self.weightMatrices[2 * i], self.weightMatrices[2 * i] - self.learningRate * totalGradW_h) )
            updates.append( (self.weightMatrices[2 * i + 1], self.weightMatrices[2 * i + 1] - self.learningRate * totalGradW_i) )
        updates.append( (self.weightMatrices[2 * self.bpttOrder], self.weightMatrices[2 * self.bpttOrder] - self.learningRate * T.grad(cost, self.weightMatrices[2 * self.bpttOrder])) )

        if (index == self.bpttOrder - 1):
            updates.append( (self.lastHiddenOut, lastHiddenOutUpdate) )
        elif (sntncEnd):
            updates.append( (self.lastHiddenOut, self.initLastHiddenOut()) )
        return updates

    def updateScan(self, cost, lastHiddenOutUpdate, sntncMiddle, sntncEnd):
        params = [ self.weightMatrices[0], self.weightMatrices[1], self.weightMatrices[2] ]
        gparams = [T.grad(cost, param) for param in params]
        updates = [(param, param - self.learningRate * gparam) for param, gparam in zip(params, gparams)]
        if (sntncEnd):
            updates.append( (self.lastHiddenOut, self.initLastHiddenOut()) )
        elif (sntncMiddle):
            updates.append( (self.lastHiddenOut, lastHiddenOutUpdate) )
        return updates

    def updateTest(self, lastHiddenOutUpdate, sntncEnd, batchSize):
        updates = []
        if (sntncEnd):
            updates.append( (self.lastHiddenOut, self.initLastHiddenOut(batchSize)) )
        else:
            updates.append( (self.lastHiddenOut, lastHiddenOutUpdate) )
        return updates

    def calculateError(self, trainFeats, trainLabels):
        batchNum = 7   #1124823 = 3*7*29*1847
        calcErrorSize = len(trainFeats) / batchNum
        forwardFunction = self.getForwardFunction(trainFeats, calcErrorSize, self.weightMatrices)
        self.errorNum = 0
        for i in xrange(batchNum):
            forwardOutput = forwardFunction(i)
            self.errorNum += np.sum(T.argmax(forwardOutput, 0).eval() != wordUtil.labelsToIndices(trainLabels[i*calcErrorSize:(i+1)*calcErrorSize]))
        self.errorRate = self.errorNum / float(calcErrorSize * batchNum)

    def initLastHiddenOut(self, batchSize=None):
        # return shared( np.zeros( (self.neuronNumList[0][0], 1), dtype=theano.config.floatX ) )
        if (batchSize is None):
            batchSize = self.batchSize
        # return shared( np.zeros( (self.neuronNumList[0][0], batchSize), dtype=theano.config.floatX ) )
        return shared( np.zeros( (batchSize, self.neuronNumList[0][0]), dtype=theano.config.floatX ) )

    ### Model generate, save and load ###
    def setRandomModel(self):
        # w_h = np.asarray( np.random.normal(
        #     loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[1]),
        #     size=(self.neuronNumList[1], self.neuronNumList[0][0])), dtype=theano.config.floatX)
        # w_i = np.asarray( np.random.normal(
        #     loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[1]),
        #     size=(self.neuronNumList[1], self.neuronNumList[0][1])), dtype=theano.config.floatX)
        # w_o = np.asarray( np.random.normal(
        #     loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[2]),
        #     size=(self.neuronNumList[2], self.neuronNumList[1])), dtype=theano.config.floatX)
        w_h = np.asarray( np.random.normal(
            loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[1]),
            size=(self.neuronNumList[0][0], self.neuronNumList[1])), dtype=theano.config.floatX)
        w_i = np.asarray( np.random.normal(
            loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[1]),
            size=(self.neuronNumList[0][1], self.neuronNumList[1])), dtype=theano.config.floatX)
        # w_o = np.asarray( np.random.normal(
        #     loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[2]),
        #     size=(self.neuronNumList[1], self.neuronNumList[2])), dtype=theano.config.floatX)
        w_o = np.asarray( np.random.normal(
            loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[2][0]),
            size=(self.neuronNumList[2][1], self.neuronNumList[1], self.neuronNumList[2][0])), dtype=theano.config.floatX)
        w_c = np.asarray( np.random.normal(
            loc=0.0, scale=1.0/np.sqrt(self.neuronNumList[2][1]),
            size=(self.neuronNumList[1], self.neuronNumList[2][1])), dtype=theano.config.floatX)
        # for i in range( self.bpttOrder ):    #ex: range(5-1) => 0, 1, 2, 3
        self.weightMatrices.append( shared(w_h) )
        self.weightMatrices.append( shared(w_i) )
        self.weightMatrices.append( shared(w_o) )
        self.weightMatrices.append( shared(w_c) )

    def saveModel(self, SAVE_MODEL_FILENAME):
        with open(SAVE_MODEL_FILENAME, 'w') as outputModelFile:
            for i in xrange(len(self.weightMatrices)):
                # Saving weight matrices
                weightMatrix = np.asarray(self.weightMatrices[i].get_value(borrow=True, return_internal_type=True))
                weightMatrixDim = weightMatrix.shape  # Shape (matrix height, matrix width)
                outputModelFile.write(str(weightMatrix.ndim))
                if (weightMatrix.ndim == 3):
                    outputModelFile.write(' ' + str(weightMatrixDim))
                outputModelFile.write('\n')
                if (weightMatrix.ndim == 3):
                    for line in xrange(weightMatrixDim[0]): 
                        for row in xrange(weightMatrixDim[1]):
                            for col in xrange(weightMatrixDim[2]):
                                outputModelFile.write(str(weightMatrix[line][row][col]) + ' ')
                            outputModelFile.write('\n')
                        outputModelFile.write('\n')
                else:                    
                    for row in xrange(weightMatrixDim[0]):
                        for col in xrange(weightMatrixDim[1]):
                            outputModelFile.write(str(weightMatrix[row][col]) + ' ')
                        outputModelFile.write('\n')
                    outputModelFile.write('\n')

    def loadModel(self, LOAD_MODEL_FILENAME):
        print 'Loading Neural Network Model...'
        t0 = time.time()
        with open(LOAD_MODEL_FILENAME) as modelFile:
            weightMatrix = []
            lines = [line for line in modelFile]    #modelFile.readlines()
            lineIndex = 0
            while lineIndex < len(lines):
                line = lines[lineIndex]
                rowList = line.rstrip().split(" ")
                # print int(rowList[0])
                # lineIndex += 1
                if int(rowList[0]) == 3:
                    weightSubMatrix = []
                    for i in xrange(int(rowList[1][1:-1])):
                        lineIndex += 1
                        line = lines[lineIndex]
                        while line != '\n':
                            rowList = line.rstrip().split(" ")
                            weightSubMatrix.append([float(ele) for ele in rowList])
                            lineIndex += 1
                            line = lines[lineIndex]
                        weightMatrix.append(weightSubMatrix)
                        weightSubMatrix = []
                    self.weightMatrices.append(shared(np.asarray(weightMatrix)))
                    weightMatrix = []
                else:
                    lineIndex += 1
                    line = lines[lineIndex]
                    while line != '\n':
                        rowList = line.rstrip().split(" ")
                        weightMatrix.append([float(ele) for ele in rowList])
                        lineIndex += 1
                        line = lines[lineIndex]
                    self.weightMatrices.append(shared(np.asarray(weightMatrix)))
                    weightMatrix = []
                lineIndex += 1
            line = modelFile.readline()

        t1 = time.time()
        print '...costs ', t1 - t0, ' seconds'
