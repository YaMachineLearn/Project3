import re

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
    pattern = '(^\d\w\s)([\w|\s]+)( \[\w+)([\w|\s]+)'
    m = re.match(pattern, testStr)
    question = m.group(2) + m.group(4)
    answer = m.group(3)

    questionWords = question.split(' ')
    return (questionWords, answer[2:])