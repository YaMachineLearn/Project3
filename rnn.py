import theano
from theano import tensor as T
from theano import shared
from theano import function
import numpy as np
import sys
import wordUtil

class RNN(object):
    def __init__(self, numInput, numHidden, numOutput, batchSize, params=None):
        self.numInput = numInput
        self.numHidden = numHidden
        self.numOutput = numOutput
        self.batchSize = batchSize

        # Declaring Theano Symbols
        self.Input = T.tensor3('Input')  # Input (for training / testing)
        self.TrainTarget = T.ivector('TrainTarget')  # Output reference (for training)
        self.TestTarget = T.imatrix('TestTarget')  # Output reference (for testing)
        self.Hidden = T.matrix('Hidden')  # Initial hidden state (for training / testing)
        self.LogProbability = T.vector('LogProbability')  # Initial log probability (for testing)
        self.LearningRate = T.scalar('LearningRate')  # Learning rate (for training)
        self.BatchIndex = T.iscalar('BatchIndex')  # Batch Index (for training / testing)
        self.WordIndex = T.iscalar('WordIndex')  # Word Index (for training)

        # Initializing RNN Parameters
        self.params = self.loadParameters(params) if params else self.genRandomParameters()
        self.hidden = self.initHidden(batchSize)
        self.logProbability = self.initLogProbability(batchSize)

    def loadParameters(self, params):
        W_in = shared(np.asarray(params[0]).astype(theano.config.floatX), name='W_in')
        W_h = shared(np.asarray(params[1]).astype(theano.config.floatX), name='W_h')
        W_o = shared(np.asarray(params[2]).astype(theano.config.floatX), name='W_o')
        return [W_in, W_h, W_o]

    def genRandomParameters(self):
        W_in = shared(np.random.uniform(-.01, .01, (self.numInput, self.numHidden)).astype(theano.config.floatX), name='W_in')
        W_h = shared(np.random.uniform(-.01, .01, (self.numHidden, self.numHidden)).astype(theano.config.floatX), name='W_h')
        W_o = shared(np.random.uniform(-.01, .01, (self.numHidden, self.numOutput)).astype(theano.config.floatX), name='W_o')
        return [W_in, W_h, W_o]

    def trainRecurrence(self, Input_t, Hidden_tm1, *params):
        W_in = params[0]
        W_h = params[1]

        Hidden_t = T.nnet.sigmoid(T.dot(Input_t, W_in) + T.dot(Hidden_tm1, W_h))
        return Hidden_t

    def testRecurrence(self, Input_t, Target_t, Hidden_tm1, Prob_tm1, *params):
        W_in = params[0]
        W_h = params[1]
        W_o = params[2]

        Hidden_t = T.nnet.sigmoid(T.dot(Input_t, W_in) + T.dot(Hidden_tm1, W_h))
        Output_t = T.nnet.softmax(T.dot(Hidden_t, W_o))
        Prob_t = Prob_tm1 + T.log(Output_t[T.arange(Target_t.shape[0]), Target_t])
        return [Hidden_t, Prob_t]

    def initHidden(self, batchSize):
        return shared(np.zeros((batchSize, self.numHidden), dtype=theano.config.floatX), name='hidden')

    def initLogProbability(self, batchSize):
        return shared(np.zeros((batchSize), dtype=theano.config.floatX), name='logProbability')

    def paddingDataList(self, dataList):
        totalSentenceNum = len(dataList)
        totalSentenceLengths = []
        maxSentenceLength = 1
        for sentenceIndex in xrange(totalSentenceNum):
            sentenceLength = len(dataList[sentenceIndex])
            totalSentenceLengths.append(sentenceLength)
            if (sentenceLength > maxSentenceLength):
                maxSentenceLength = sentenceLength

        for sentenceIndex in xrange(totalSentenceNum):
            dataList[sentenceIndex].extend([0] * (maxSentenceLength - totalSentenceLengths[sentenceIndex]))

        return (dataList, totalSentenceLengths)

    def buildTrainFunction(self, trainDataList, bpttOrder):
        print >> sys.stderr, 'Compiling train function...'
        # Generating shared arrays for train data list
        print >> sys.stderr, '  generating shared train data...'
        (paddedTrainDataList, trainSentenceLengths) = self.paddingDataList(trainDataList)
        trainDataShared = shared(np.asarray(paddedTrainDataList, dtype=np.int32))
        trainSentenceLengthsShared = shared(np.asarray(trainSentenceLengths, dtype=np.int32))

        # Getting the length of a sentence in the batch
        sentenceLength = trainSentenceLengthsShared[self.BatchIndex * self.batchSize]
        startWordIndex = T.maximum(0, self.WordIndex - bpttOrder + 1)

        # Theano scan
        hiddenStates, _ = theano.scan(fn=self.trainRecurrence,
                                      sequences=self.Input,
                                      outputs_info=self.Hidden,
                                      non_sequences=self.params)

        # Compute network output and cost
        lastHidden = hiddenStates[-1]
        output = T.nnet.softmax(T.dot(lastHidden, self.params[2]))
        probOutputGivenInput = output[T.arange(self.TrainTarget.shape[0]), self.TrainTarget]
        cost = -T.mean(T.log(probOutputGivenInput))

        # Updates for each train function call
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - self.LearningRate * gparam) for param, gparam in zip(self.params, gparams)]
        if T.gt(startWordIndex, 0):
            updates.append((self.hidden, lastHidden) if T.lt(self.WordIndex, sentenceLength - 1) else (self.hidden, self.initHidden(self.batchSize)))

        # Getting word vectors from train data indices
        trainInputData = wordUtil.WORD_VECTORS[trainDataShared[self.BatchIndex * self.batchSize: (self.BatchIndex + 1) * self.batchSize, startWordIndex: self.WordIndex + 1]].dimshuffle(1, 0, 2)
        trainTarget = trainDataShared[self.BatchIndex * self.batchSize: (self.BatchIndex + 1) * self.batchSize, self.WordIndex + 1]

        givens = [
            (self.Input, trainInputData),
            (self.TrainTarget, trainTarget),
            (self.Hidden, self.hidden),
        ]

        print >> sys.stderr, '  generating train function...'
        trainFunction = function(inputs=[self.BatchIndex, self.WordIndex, self.LearningRate],
                                 outputs=[probOutputGivenInput, cost],
                                 updates=updates,
                                 givens=givens)
        return trainFunction

    def buildTestFunction(self, testDataList, numOfChoices):
        print >> sys.stderr, 'Compiling test function...'
        # Generating shared arrays for test data list
        print >> sys.stderr, '  generating shared test data...'
        (paddedTestDataList, testSentenceLengths) = self.paddingDataList(testDataList)
        testDataShared = shared(np.asarray(paddedTestDataList, dtype=np.int32))
        testSentenceLengthsShared = shared(np.asarray(testSentenceLengths, dtype=np.int32))

        # Getting the length of a sentence in the batch
        sentenceLength = testSentenceLengthsShared[self.BatchIndex * self.batchSize]

        # Theano scan
        [hiddenStates, sentenceLogProbability], _ = theano.scan(fn=self.testRecurrence,
                                                                sequences=[self.Input, self.TestTarget],
                                                                outputs_info=[self.Hidden, self.LogProbability],
                                                                non_sequences=self.params)

        # Computing network output
        answerChoices = T.argmax(sentenceLogProbability[-1].reshape((self.batchSize / numOfChoices, numOfChoices)), axis=1)

        # Getting word vectors from test data indices
        testInputData = wordUtil.WORD_VECTORS[testDataShared[self.BatchIndex * self.batchSize: (self.BatchIndex + 1) * self.batchSize, 0: sentenceLength - 1]].dimshuffle(1, 0, 2)
        testTarget = testDataShared[self.BatchIndex * self.batchSize: (self.BatchIndex + 1) * self.batchSize, 1: sentenceLength].T

        givens = [
            (self.Input, testInputData),
            (self.TestTarget, testTarget),
            (self.Hidden, self.initHidden(self.batchSize)),
            (self.LogProbability, self.initLogProbability(self.batchSize))
        ]

        print >> sys.stderr, '  generating test function...'
        testFunction = function(inputs=[self.BatchIndex],
                                outputs=[answerChoices, sentenceLogProbability[-1]],
                                givens=givens)
        return testFunction

    def getParams(self):
        return [param.get_value() for param in self.params]
