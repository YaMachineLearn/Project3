from parse import parseWordVectors
import numpy as np

WORD_VECTORS_FILENAME = "data/vec_v5.txt"
OTHER_TYPE_SYMBOL = "@OTHER@"  # The symbol representing the OTHER type of word

parsedWordVectors, parsedWords = parseWordVectors(WORD_VECTORS_FILENAME)
VECTOR_DIM = len(parsedWordVectors[0])

# Adding the OTHER type of word to words and word vectors (OTHER -> [0.0] * WORD_COUNT)
parsedWordVectors.append([0.] * VECTOR_DIM)
parsedWords.append(OTHER_TYPE_SYMBOL)

# Removing </s>
# WORD_VECTORS = shared( np.asarray(parsedWordVectors[1:], dtype=theano.config.floatX) )
# WORDS = parsedWords[1:]

WORD_VECTORS = np.asarray(parsedWordVectors)
WORDS = parsedWords

TOTAL_WORDS = len(WORDS)  # Number of words including the OTHER type

# del parsedWordVectors, parsedWords

# Making dictionaries for word <-> word index mapping
WORD_INDEX_DICT = dict(zip(WORDS, range(TOTAL_WORDS)))

OTHER_WORD_INDEX = WORD_INDEX_DICT[OTHER_TYPE_SYMBOL]

def wordToindex(word):
    return WORD_INDEX_DICT.get(word, OTHER_WORD_INDEX)

def wordSimilarity(wordIndex1, wordIndex2):
    return np.dot(WORD_VECTORS[wordIndex1], WORD_VECTORS[wordIndex2])
