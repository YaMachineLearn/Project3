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
testWordsChecklist = [False] * wordUtil.TOTAL_WORDS

print 'Checking words in testing data...'
t0 = time.time()
for sentence in testWordIndices:
    for wordIndex in sentence:
        testWordsChecklist[wordIndex] = True
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Calculating word frequencies in training data...'
t0 = time.time()
trainWordsHist = np.bincount(np.asarray([wordIndex for sentence in trainWordIndices for wordIndex in sentence], dtype='int32'))
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Extracting useful words...'
t0 = time.time()
newWordsIndices = [index for index in range(len(testWordsChecklist)) if wordUtil.WORDS[index] != wordUtil.OTHER_TYPE_SYMBOL and (testWordsChecklist[index] or trainWordsHist[index] >= WORD_FREQUENCY_THRES)]
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'

print 'Extracted', len(newWordsIndices), 'words (without OTHER and <s>)'

print 'Outputing new word vectors to', OUTPUT_VEC_FILENAME
t0 = time.time()
with open(OUTPUT_VEC_FILENAME, 'w') as outputVecFile:
    # Writing header
    outputVecFile.write(str(len(newWordsIndices) + 2) + ' ' + str(wordUtil.WORD_VECTOR_SIZE) + '\n')
    # Writing </s> and <s>
    for i in range(2):
        outputVecFile.write(wordUtil.parsedWords[i] + ' ')
        for val in wordUtil.parsedWordVectors[i]:
            outputVecFile.write(str(val) + ' ')
        outputVecFile.write('\n')

    # Writing new vectors
    wordVectors = wordUtil.WORD_VECTORS.get_value()
    for i in range(len(newWordsIndices)):
        outputVecFile.write(wordUtil.WORDS[newWordsIndices[i]] + ' ')
        for val in wordVectors[newWordsIndices[i]]:
            outputVecFile.write(str(val) + ' ')
        outputVecFile.write('\n')
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'