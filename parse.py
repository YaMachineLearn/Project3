import re
import math
import time

SENTENCE_LENGTH_THRES = 64

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

import wordUtil  # Must be imported after parseWordVectors has been defined for labelUtil to use

# DATA_FILENAME includes TRAIN_FILE_NAME or PROBLEM_FILE_NAME
def parseData(DATA_FILENAME):
    dataWordIndices = []

    with open(DATA_FILENAME) as trainFeatFile:
        for line in trainFeatFile:
            strippedLine = line.rstrip()
            if strippedLine:   #not empty after strip
                lineList = strippedLine.split(' ')
                if len(lineList) < SENTENCE_LENGTH_THRES:
                    oneLineWordIndices = [wordUtil.wordToindex(word) for word in lineList]
                    dataWordIndices.append(oneLineWordIndices)
                
    return dataWordIndices
"""
def getProblemAndAnswer(testStr):
    pattern = '(^\d+\w\s)([\w|\s]+)( \[\w+)([\w|\s]*[^\n])(\n?)$'
    m = re.match(pattern, testStr)
    problem = m.group(2) + m.group(4) if m.group(4) != '' else m.group(2)
    answer = m.group(3)

    problemWords = problem.split(' ')
    return (problemWords, answer[2:])

def parseProblemsAndAnswers(TEST_FILENAME):
    problemSet = []
    answersSet = []
    with open(TEST_FILENAME) as testFeatFile:
        lineNum = 1
        oneProblem = []
        oneAnswers = []
        for line in testFeatFile:
            problemWords, answer = getProblemAndAnswer(line)
            answerWordVec = labelUtil.wordToVector(answer)
            oneAnswers.append(answerWordVec)
            if lineNum % 5 == 0:
                for problemWord in problemWords:
                    problemWordVec = labelUtil.wordToVector(problemWord)
                    oneProblem.append(problemWordVec)
                problemSet.append(oneProblem)
                answersSet.append(oneAnswers)
                oneProblem = []
                oneAnswers = []
            lineNum += 1
    return (problemSet, answersSet)
"""
def outputCsvFileFromAnswerNumbers(guessAnswer, OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w') as outputFile:
        outputFile.write('Id,Answer\n')
        for i in xrange(1040):
            outputFile.write(str(i + 1) + ',' + chr(97 + guessAnswer[i]) + '\n' )

def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    val = min(1, max(dotproduct(v1, v2) / (length(v1) * length(v2)), -1))
    return math.acos(val)