import re
import numpy as np
import math

def parseWordVectors(VEC_FILENAME):
    trainFeats = []
    trainLabels = []

    #parse training features
    with open(VEC_FILENAME) as trainFeatFile:
        next(trainFeatFile)
        for line in trainFeatFile:
            strippedLine = line.rstrip()
            if strippedLine:   #not empty after strip
                lineList = strippedLine.split(' ')
                trainLabels.append(lineList.pop(0))
                trainFeats.append([float(ele) for ele in lineList])

    return (trainFeats, trainLabels)

def getProblemAndAnswer(testStr):
    pattern = '(^\d+\w\s)([\w|\s]+)( \[\w+)([\w|\s]*[^\n])(\n?)$'
    m = re.match(pattern, testStr)
    problem = m.group(2) + m.group(4) if m.group(4) != '' else m.group(2)
    answer = m.group(3)

    problemWords = problem.split(' ')
    return (problemWords, answer[2:])

def parseTestFeatures(TEST_FILENAME, trainFeats, trainLabels):
    problemSet = []
    answersSet = []
    with open(TEST_FILENAME) as testFeatFile:
        lineNum = 1
        oneProblem = []
        oneAnswers = []
        for line in testFeatFile:
            problemWords, answer = getProblemAndAnswer(line)
            answerWordVec = trainFeats[trainLabels.index(answer)] if answer in trainLabels else np.zeros([1, 300], dtype=np.float32).flatten()
            oneAnswers.append(answerWordVec)
            if lineNum % 5 == 0:
                for problemWord in problemWords:
                    problemWordVec = trainFeats[trainLabels.index(problemWord)] if problemWord in trainLabels else np.zeros([1, 300], dtype=np.float32).flatten()
                    oneProblem.append(problemWordVec)
                problemSet.append(oneProblem)
                answersSet.append(oneAnswers)
                oneProblem = []
                oneAnswers = []
            lineNum += 1
    return (problemSet, answersSet)

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