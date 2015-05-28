import random
import wordUtil

SENTENCE_LENGTH_THRES = 64

def parseWordVectors(VEC_FILENAME):
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

    return (wordVectors, words)

# DATA_FILENAME includes TRAIN_FILE_NAME or PROBLEM_FILE_NAME
def parseData(DATA_FILENAME):
    dataWordIndices = []

    with open(DATA_FILENAME) as trainFeatFile:
        for line in trainFeatFile:
            strippedLine = line.rstrip()
            if strippedLine:   #not empty after strip
                lineList = strippedLine.split(' ')
                if len(lineList) < SENTENCE_LENGTH_THRES:
                    oneLineWordIndices = [wordUtil.wordToIndex(word) for word in lineList]
                    dataWordIndices.append(oneLineWordIndices)
                
    return dataWordIndices

def parseAndClusterTrainData(TRAIN_FILE, BATCH_SIZE, SNTNC_LEN_THRES=SENTENCE_LENGTH_THRES):
    # sample usage: parseAndClusterTrainData(TRAIN_FILENAME, 64, 256)
    # output: 3-dim list,
    #     form: [ [sentences with len = 0], [sentences with len = 1], ..., sentences with len = 63 ]
    #     ex: [ [ [23,845,568], [4896,4165,159,852,198] ], [[...],[...],[...],[...]], ... ]

    # read file, and convert words into word indices
    sentences = list()
    with open(TRAIN_FILE) as trainFile:
        for line in trainFile:
            strippedLine = line.rstrip()
            if strippedLine:   #not empty after strip
                lineList = strippedLine.split(' ')
                sntncLen = len(lineList)
                if sntncLen < SNTNC_LEN_THRES:
                    oneLineWordIndices = [wordUtil.wordToIndex(word) for word in lineList]
                    sentences.append(oneLineWordIndices)

    # shuffle sentences by shuffle the index
    # because we will discard part of the data
    # because we want len of each cluster be BATCH_SIZE*n
    shuffledIndex = range(len(sentences))
    random.shuffle(shuffledIndex)

    #cluster sentences
    clusteredSentences = list()
    for i in xrange(SNTNC_LEN_THRES):
        clusteredSentences.append(list())

    clusterCount = len(clusteredSentences)
    for sentence in sentences:
        sntncLen = len(sentence)
        if sntncLen < clusterCount:
            clusteredSentences[sntncLen].append(sentence)

    # make the len of each cluster % BATCH_SIZE == 0
    for cluster in clusteredSentences:
        redundentSntncCount = len(cluster) % BATCH_SIZE
        for i in xrange(redundentSntncCount):
            cluster.pop()

    dataWordIndices = list()

    # output new data word indices
    for cluster in clusteredSentences:
        for wordIndices in cluster:
            dataWordIndices.append(wordIndices)

    return dataWordIndices

def outputCsvFileFromAnswerNumbers(guessAnswers, OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w') as outputFile:
        outputFile.write('Id,Answer\n')
        for i in xrange(len(guessAnswers)):
            outputFile.write(str(i + 1) + ',' + chr(97 + guessAnswers[i]) + '\n' )