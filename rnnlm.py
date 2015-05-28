from rnn import RNN
import numpy as np
import sys
import time

class RNNLM(object):
    def __init__(self, neuronNumList, SAVE_MODEL_FILENAME=None, LOAD_MODEL_FILENAME=None):
        self.neuronNumList = neuronNumList

        # Model saving / loading and relative parameters
        self.saveModelFilename = SAVE_MODEL_FILENAME
        self.loadModelFilename = LOAD_MODEL_FILENAME
        self.params = None

        # RNN Models for training / testing
        self.trainRNN = None
        self.testRNN = None

    def train(self, epochNum, batchSize, bpttOrder, learningRate, trainData):
        print >> sys.stderr, 'Building RNN model for training...'
        self.params = self.loadModel()
        self.trainRNN = RNN(self.neuronNumList[0][1], self.neuronNumList[1], self.neuronNumList[2][0], batchSize, self.params)
        trainFunction = self.trainRNN.buildTrainFunction(trainData, bpttOrder)

        totalTrainDataNum = len(trainData)
        batchNum = totalTrainDataNum / batchSize

        print >> sys.stderr, '  Start Training...'
        for epoch in xrange(epochNum):
            print >> sys.stderr, '  Epoch -', epoch + 1, ':'
            t0 = time.time()
            sumEpochCost = 0
            epochProgress = 0.
            for batchIndex in xrange(batchNum):
                sumBatchCost = 0
                sys.stdout.write('\r    Epoch Progress: %2.4f%%, Batch: %d/%d' % (epochProgress, batchIndex + 1, batchNum))
                sys.stdout.flush()
                t3 = time.time()

                sentenceLength = len(trainData[batchIndex * batchSize])
                for wordIndex in xrange(sentenceLength - 1):
                    probOutputGiveInput, cost = trainFunction(batchIndex, wordIndex, learningRate)
                    sumBatchCost += cost

                t4 = time.time()
                averageBatchCost = (sumBatchCost / (sentenceLength - 1))
                sumEpochCost += averageBatchCost
                epochProgress = float(batchIndex + 1) / float(batchNum) * 100.
                sys.stdout.write('\r    Epoch Progress: %2.4f%%, Batch: %d/%d, Avg batch cost: %2.4f, Cost: %2.1fs' % (epochProgress, batchIndex + 1, batchNum, averageBatchCost, t4 - t3))
                sys.stdout.flush()
            t1 = time.time()
            print '  ...Average epoch cost per batch:', sumEpochCost / batchNum
            print '  ...costs', t1 - t0, 'seconds'

        self.saveModel()

    def test(self, batchSize, testData):
        print >> sys.stderr, 'Building RNN model for testing...'
        self.params = self.loadModel()
        if self.params == None:
            print >> sys.stderr, '- Warning: running test without any trained models'
        self.testRNN = RNN(self.neuronNumList[0][1], self.neuronNumList[1], self.neuronNumList[2][0], batchSize, self.params)
        testFunction = self.testRNN.buildTestFunction(testData)

        totalTestDataNum = len(testData)
        batchNum = totalTestDataNum / batchSize
        answers = []

        print >> sys.stderr, '  Start Testing...'
        t0 = time.time()
        progress = 0.
        for batchIndex in xrange(batchNum):
            sys.stdout.write('\r    Progress: %2.4f%%, Batch: %d/%d' % (progress, batchIndex + 1, batchNum))
            sys.stdout.flush()
            t3 = time.time()

            [answerChoice, logProbability] = testFunction(batchIndex)
            answers.append(answerChoice.item())  # Answer choice is a numpy 0-d array, using .item() to get the value
            # print logProbability

            t4 = time.time()
            progress = float(batchIndex + 1) / float(batchNum) * 100.
            sys.stdout.write('\r    Progress: %2.4f%%, Batch: %d/%d, Cost: %2.1fs' % (progress, batchIndex + 1, batchNum, t4 - t3))
            sys.stdout.flush()
        t1 = time.time()
        print
        print '  ...costs ', t1 - t0, ' seconds'
        return answers

    def saveModel(self):
        if self.saveModelFilename:
            if self.trainRNN:
                print >> sys.stderr, '- Saving RNN model parameters to', self.saveModelFilename, '...'
                params = self.trainRNN.getParams()
                np.savez(self.saveModelFilename, *params)
            else:
                print >> sys.stderr, '- Error: Nothing trained yet! You have to train first before saving model!'

    def loadModel(self):
        if self.trainRNN:
            print >> sys.stderr, '- Loading RNN model parameters from previously trained RNN model...'
            return self.trainRNN.getParams()
        elif self.loadModelFilename:
            print >> sys.stderr, '- Loading RNN model parameters from', self.loadModelFilename, '...'
            npzFile = np.load(self.loadModelFilename + '.npz')
            return [npzFile[param] for param in sorted(npzFile.files)]
        else:
            print >> sys.stderr, '- Nothing to load'
            return None
