import re
import numpy as np

def parseTrainFeatures(VEC_FILENAME):
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
