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
        self.neuronNumList = neuronNumList
        self.bpttOrder = bpttOrder
        self.learningRate = learningRate
        self.epochNum = epochNum
        self.batchSize = batchSize

