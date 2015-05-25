import parse
import numpy as np
import theano
from theano import shared

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

# Making dictionaries for word <-> word index mapping
WORD_INDEX_DICT = dict(zip(WORDS, range(TOTAL_WORDS)))

def wordToindex(word):
    return WORD_INDEX_DICT.get(word, WORD_INDEX_DICT[OTHER_TYPE_SYMBOL])