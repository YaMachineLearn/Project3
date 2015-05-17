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

def getQuestionAndAnswer(testStr):
    pattern = '(^\d+\w\s)([\w|\s]+)( \[\w+)([\w|\s]*[^\n])(\n?)$'
    m = re.match(pattern, testStr)
    question = m.group(2) + m.group(4) if m.group(4) != '' else m.group(2)
    answer = m.group(3)

    questionWords = question.split(' ')
    return (questionWords, answer[2:])

def parseTestFeatures(TEST_FILENAME, trainFeats, trainLabels):
    problemSet = []
    answersSet = []
    with open(TEST_FILENAME) as testFeatFile:
        lineNum = 0
        oneProblem = []
        oneAnswers = []
        for line in testFeatFile:
            questionWords, answer = getQuestionAndAnswer(line)
            if lineNum % 5 == 0:
                if lineNum != 0:
                    problemSet.append(oneProblem)
                    answersSet.append(oneAnswers)
                oneProblem = []
                oneAnswers = []
                for i in xrange(len(questionWords)):
                    problemWordVec = trainFeats[trainLabels.index(questionWords[i])] if questionWords[i] in trainLabels else np.zeros([1, 300], dtype=np.float32).flatten()
                    answerWordVec = trainFeats[trainLabels.index(answer)] if answer in trainLabels else np.zeros([1, 300], dtype=np.float32).flatten()
                    oneProblem.append(problemWordVec)
                    oneAnswers.append(answerWordVec)
                ++lineNum
            else:
                oneAnswers.append(trainFeats[trainLabels.index(answer)])
                ++lineNum
        problemSet.append(oneProblem)
        answersSet.append(oneAnswers)
    return (problemSet, answersSet)
