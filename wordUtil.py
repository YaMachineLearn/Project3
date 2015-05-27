import parse
import numpy as np
from math import ceil
import theano
from theano import shared
# TOTAL_WORDS = 8
# WORD_VECTOR_SIZE = 8
WORD_CLASS_NUM = 64 # number of classes
WORD_CLASS_SIZE = None # will be size of a class
# WORD_CLASSES = shared( np.asarray([1, 1, 1, 0, 0, 1, 0, 0], dtype='int32') )
# WORD_CLASS_LABELS = shared( np.asarray([0, 1, 2, 0, 1, 3, 2, 3], dtype='int32') )
# WORD_VECTORS = shared( np.asarray([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]], dtype='int32') )

WORD_VECTORS_FILENAME = "data/vec_reduced.txt"

parsedWordVectors, parsedWords = parse.parseWordVectors(WORD_VECTORS_FILENAME)

WORD_VECTOR_SIZE = len(parsedWordVectors[0])

OTHER_TYPE_SYMBOL = "@OTHER@"  # The symbol representing the OTHER type of word

# Adding the OTHER type of word to words and word vectors (OTHER -> [0.0] * WORD_VECTOR_SIZE)
parsedWordVectors.append([0.] * WORD_VECTOR_SIZE)
parsedWords.append(OTHER_TYPE_SYMBOL)

# Removing </s>
WORD_VECTORS = shared( np.asarray(parsedWordVectors[1:], dtype=theano.config.floatX) )
WORDS = parsedWords[1:]

TOTAL_WORDS = len(WORDS)  # Number of words including the OTHER type
WORD_CLASS_SIZE = int(ceil(float(TOTAL_WORDS) / float(WORD_CLASS_NUM))) # size of a class

# Making dictionaries for word <-> word index mapping
WORD_INDEX_DICT = dict(zip(WORDS, range(TOTAL_WORDS)))

def wordToindex(word):
    return WORD_INDEX_DICT.get(word, WORD_INDEX_DICT[OTHER_TYPE_SYMBOL])

def genWordClassUtils(trainLabels): 
    hist = np.bincount(np.asarray([item for sublist in trainLabels for item in sublist], dtype='int32')) 
    totalWords = np.sum(hist) 
    orderedIndices = np.argsort(hist) 
    WORD_CLASSES_list = [-1] * TOTAL_WORDS 
    WORD_CLASS_LABELS_list = [-1] * TOTAL_WORDS 
    for i in xrange(TOTAL_WORDS): 
        WORD_CLASSES_list[orderedIndices[TOTAL_WORDS - 1 - i]] = i / WORD_CLASS_SIZE
        WORD_CLASS_LABELS_list[orderedIndices[TOTAL_WORDS - 1 - i]] = i % WORD_CLASS_SIZE
    global WORD_CLASSES 
    global WORD_CLASS_LABELS 
    WORD_CLASSES = shared( np.asarray(WORD_CLASSES_list, dtype='int32') ) 
    WORD_CLASS_LABELS = shared( np.asarray(WORD_CLASS_LABELS_list, dtype='int32') ) 

    saveWordClassUtilsModel("models/wordClass.dat", WORD_CLASSES_list, WORD_CLASS_LABELS_list)

def saveWordClassUtilsModel(WORD_CLASS_FILE, WORD_CLASSES_list, WORD_CLASS_LABELS_list):
    with open(WORD_CLASS_FILE, 'w') as wordClassFile:
        for ele in WORD_CLASSES_list:
            wordClassFile.write(str(ele))
            wordClassFile.write(' ')
        wordClassFile.write('\n')
        for ele in WORD_CLASS_LABELS_list:
            wordClassFile.write(str(ele))
            wordClassFile.write(' ')
        wordClassFile.write('\n')

def loadWordClassUtilsModel(WORD_CLASS_FILE):
    global WORD_CLASSES 
    global WORD_CLASS_LABELS 

    with open(WORD_CLASS_FILE) as wordClassFile:
        line1 = wordClassFile.readline()
        WORD_CLASSES_list = [int(ele) for ele in line1.split()]
        WORD_CLASSES = shared( np.asarray(WORD_CLASSES_list, dtype='int32') ) 

        line2 = wordClassFile.readline()
        WORD_CLASS_LABELS_list = [int(ele) for ele in line2.split()]
        WORD_CLASS_LABELS = shared( np.asarray(WORD_CLASS_LABELS_list, dtype='int32') ) 
