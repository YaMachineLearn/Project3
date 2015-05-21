import time

WORD_VECTORS_FILENAME = "vec.txt"

def parseWordVectors(VEC_FILENAME):
    print 'Parsing word vectors...'
    t0 = time.time()

    wordVectors = []
    words = []

    #parse word vectors
    with open(VEC_FILENAME) as wordVecFile:
        next(wordVecFile)
        for line in wordVecFile:
            strippedLine = line.rstrip()
            if strippedLine:   #not empty after strip
                lineList = strippedLine.split(' ')
                words.append(lineList.pop(0))
                wordVectors.append([float(ele) for ele in lineList])

    t1 = time.time()
    print '...costs ', t1 - t0, ' seconds'
    return (wordVectors, words)

parsedWordVectors, parsedWords = parseWordVectors(WORD_VECTORS_FILENAME)

OTHER_TYPE_SYMBOL = "@OTHER@"  # The symbol representing the OTHER type of word

# Removing <s> and </s>
WORD_VECTORS = parsedWordVectors[2:-1]
WORDS = parsedWords[2:-1]

# Adding the OTHER type of word to words and word vectors (OTHER -> [0.0] * 300)
WORD_VECTORS.append([0.] * 300)
WORDS.append(OTHER_TYPE_SYMBOL)

TOTAL_WORDS = len(WORDS)  # Number of words including the OTHER type

del parsedWordVectors, parsedWords

# Making dictionaries for word <-> word vector / word <-> word index mapping
WORD_VECTOR_DICT = dict(zip(WORDS, WORD_VECTORS))
WORD_INDEX_DICT = dict(zip(WORDS, range(TOTAL_WORDS)))

# Getting vector/index from a random word, if the given word is not found, then the vector/index of OTHER type will be returned
def wordToVector(word):
    return WORD_VECTOR_DICT.get(word, WORD_VECTOR_DICT[OTHER_TYPE_SYMBOL])

def wordToindex(word):
    return WORD_INDEX_DICT.get(word, WORD_INDEX_DICT[OTHER_TYPE_SYMBOL])

def wordTo1ofNVector(word):
    vec1ofN = [0.] * TOTAL_WORDS
    vec1ofN[wordToindex(word)] = 1.
    return vec1ofN