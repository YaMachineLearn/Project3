import parse
import numpy as np
import theano
from theano import shared

OTHER_TYPE_SYMBOL = "@OTHER@"

def parseWordVectors(WORD_VECTORS_FILENAME):
    parsedWordVectors, parsedWords = parse.parseWordVectors(WORD_VECTORS_FILENAME)

    global WORD_VECTOR_SIZE
    global WORD_VECTORS
    global WORDS
    global TOTAL_WORDS
    global WORD_INDEX_DICT

    WORD_VECTOR_SIZE = len(parsedWordVectors[0])

    # Adding the OTHER type of word to words and word vectors (OTHER -> [0.0] * WORD_VECTOR_SIZE)
    parsedWordVectors.append([0.] * WORD_VECTOR_SIZE)
    parsedWords.append(OTHER_TYPE_SYMBOL)

    WORD_VECTORS = shared(np.asarray(parsedWordVectors, dtype=theano.config.floatX))
    WORDS = parsedWords

    TOTAL_WORDS = len(WORDS)  # Number of words including the OTHER type

    # Making dictionaries for word <-> word vector / word <-> word index mapping
    WORD_INDEX_DICT = dict(zip(WORDS, range(TOTAL_WORDS)))

def wordToIndex(word):
    return WORD_INDEX_DICT.get(word, WORD_INDEX_DICT[OTHER_TYPE_SYMBOL]) if WORD_INDEX_DICT else 0

# WORD_CLASS_NUM = 100
#
# def genWordClassUtils(trainLabels):
#     hist = np.bincount(np.asarray([item for sublist in trainLabels for item in sublist], dtype='int32'))
#     totalWords = np.sum(hist)
#     orderedIndices = np.argsort(hist)
#     index = TOTAL_WORDS - 1
#     freqSum = 0
#     WORD_CLASS_CUM_SIZES_list = [0]
#     WORD_CLASSES_list = [-1] * TOTAL_WORDS
#     WORD_CLASS_LABELS_list = [-1] * TOTAL_WORDS
#     for i in xrange(WORD_CLASS_NUM):
#         lab = 0
#         while (freqSum < totalWords * (i + 1) / WORD_CLASS_NUM):
#             freqSum += hist[orderedIndices[index]]
#             WORD_CLASSES_list[orderedIndices[index]] = i
#             WORD_CLASS_LABELS_list[orderedIndices[index]] = lab
#             index -= 1
#             lab += 1
#         WORD_CLASS_CUM_SIZES_list.append( TOTAL_WORDS - 1 - index )
#     global WORD_CLASS_CUM_SIZES
#     global WORD_CLASSES
#     global WORD_CLASS_LABELS
#     WORD_CLASS_CUM_SIZES = shared( np.asarray(WORD_CLASS_CUM_SIZES_list, dtype='int32') )
#     WORD_CLASSES = shared( np.asarray(WORD_CLASSES_list, dtype='int32') )
#     WORD_CLASS_LABELS = shared( np.asarray(WORD_CLASS_LABELS_list, dtype='int32') )