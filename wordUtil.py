import parse
import numpy as np
import theano
from theano import shared
WORD_CLASS_NUM = 100

WORD_VECTORS_FILENAME = "data/vec.txt"

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

del parsedWordVectors, parsedWords

# Making dictionaries for word <-> word vector / word <-> word index mapping
# WORD_VECTOR_DICT = dict(zip(WORDS, WORD_VECTORS))
WORD_INDEX_DICT = dict(zip(WORDS, range(TOTAL_WORDS)))

# Getting vector/index from a random word, if the given word is not found, then the vector/index of OTHER type will be returned
# def wordToVector(word):
#     return WORD_VECTOR_DICT.get(word, WORD_VECTOR_DICT[OTHER_TYPE_SYMBOL])

def wordToindex(word):
    return WORD_INDEX_DICT.get(word, WORD_INDEX_DICT[OTHER_TYPE_SYMBOL])

# def wordTo1ofNVector(word):
#     vec1ofN = [0.] * TOTAL_WORDS
#     vec1ofN[wordToindex(word)] = 1.
#     return vec1ofN

def genWordClassUtils(trainLabels):
    hist = np.bincount(np.asarray([item for sublist in trainLabels for item in sublist], dtype='int32'))
    totalWords = np.sum(hist)
    orderedIndices = np.argsort(hist)
    index = TOTAL_WORDS - 1
    freqSum = 0
    WORD_CLASS_CUM_SIZES_list = [0]
    WORD_CLASSES_list = [-1] * TOTAL_WORDS
    WORD_CLASS_LABELS_list = [-1] * TOTAL_WORDS
    for i in xrange(WORD_CLASS_NUM):
        lab = 0
        while (freqSum < totalWords * (i + 1) / WORD_CLASS_NUM):
            freqSum += hist[orderedIndices[index]]
            WORD_CLASSES_list[orderedIndices[index]] = i
            WORD_CLASS_LABELS_list[orderedIndices[index]] = lab
            index -= 1
            lab += 1
        WORD_CLASS_CUM_SIZES_list.append( TOTAL_WORDS - 1 - index )
    global WORD_CLASS_CUM_SIZES
    global WORD_CLASSES
    global WORD_CLASS_LABELS
    WORD_CLASS_CUM_SIZES = shared( np.asarray(WORD_CLASS_CUM_SIZES_list, dtype='int32') )
    WORD_CLASSES = shared( np.asarray(WORD_CLASSES_list, dtype='int32') )
    WORD_CLASS_LABELS = shared( np.asarray(WORD_CLASS_LABELS_list, dtype='int32') )