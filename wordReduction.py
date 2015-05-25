import numpy as np
import parse
import wordUtil
import time

TRAIN_FILENAME = "data/training_v4_noTag.txt"
TEST_FILENAME = "data/test_v4.txt"

OUTPUT_VEC_FILENAME = "data/vec_reduced.txt"

WORD_FREQUENCY_THRES = 1000

print 'Parsing training data...'
t0 = time.time()
trainWordIndices = parse.parseData(TRAIN_FILENAME)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Parsing testing data...'
t0 = time.time()
testWordIndices = parse.parseData(TEST_FILENAME)
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Total', wordUtil.TOTAL_WORDS - 2, 'words (without OTHER and <s>)'

# A check list for words existing in TEST_FILE
testWordsMap = dict(zip(wordUtil.WORDS[1:-1], [False] * (wordUtil.TOTAL_WORDS - 2)))  # Excluding the <s> and the OTHER type

print 'Checking words in testing data...'
t0 = time.time()
for sentence in testWordIndices:
    for wordIndex in sentence:
        testWordsMap[wordUtil.WORDS[wordIndex]] = True
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Calculating word frequencies in training data...'
t0 = time.time()
trainWordsHist = np.bincount(np.asarray([wordIndex for sentence in trainWordIndices for wordIndex in sentence], dtype='int32'))
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Extracting useful words...'
t0 = time.time()
newWords = [key for key in testWordsMap if testWordsMap[key] or trainWordsHist[wordUtil.wordToindex(key)] >= WORD_FREQUENCY_THRES]
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Extracted', len(newWords), 'words (without OTHER and <s>)'

print 'Outputing new word vectors to', OUTPUT_VEC_FILENAME
t0 = time.time()
with open(OUTPUT_VEC_FILENAME, 'w') as outputVecFile:
    # Writing header
    outputVecFile.write(str(len(newWords)) + ' ' + str(wordUtil.WORD_VECTOR_SIZE) + '\n')
    # Writing </s> and <s>
    for i in range(2):
        outputVecFile.write(wordUtil.parsedWords[i] + ' ')
        for val in wordUtil.parsedWordVectors[i]:
            outputVecFile.write(str(val) + ' ')
        outputVecFile.write('\n')

    # Writing new vectors
    wordVectors = wordUtil.WORD_VECTORS.get_value()
    for i in range(len(newWords)):
        outputVecFile.write(newWords[i] + ' ')
        wordIndex = wordUtil.wordToindex(newWords[i])
        for val in wordVectors[wordIndex]:
            outputVecFile.write(str(val) + ' ')
        outputVecFile.write('\n')
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'