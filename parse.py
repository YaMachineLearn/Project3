import re
import math
import time
import wordUtil as wu   #wordUtil must has been imported in main.py

CHOICE_NUM = 5
ANSWERS = ['a', 'b', 'c', 'd', 'e']

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

def parseProblemChoices(TEST_FILE):
    # TEST_FILE: preprocessed file. only english characters in it
    #output.choicesList: [[324,8752,4657,123,9527], ...] #choices in each problem
    #output.problemList: [[25,896,1888], ...]  #other words in each problem


    problemList = list()
    choicesList = list()

    testFile = open(TEST_FILE)
    fileLines = testFile.readlines()
    testFile.close()

    problemCount = len(fileLines) / CHOICE_NUM

    for problemIndex in xrange(problemCount):
        problemLines = fileLines[ problemIndex*CHOICE_NUM : (problemIndex+1)*CHOICE_NUM ]

        problemWordIndices = [ [wu.wordToindex(word) for word in sentence.split()] for sentence in problemLines]
        #problemWordIndices: [ [25,896,324,1888], [25,896,8752,1888], [25,896,4657,1888], [25,896,123,1888], [25,896,9527,1888]  ]

        for i in xrange(len(problemWordIndices[0])):
            if problemWordIndices[0][i] != problemWordIndices[1][i]:
                answerIndex = i
                break

        choices = [ sentence[answerIndex] for sentence in problemWordIndices ]
        #choices: [324,8752,4657,123,9527]
        problemWordIndices[0].pop(answerIndex)
        problem = problemWordIndices[0]
        #problem: [25,896,1888]

        problemList.append(problem)
        choicesList.append(choices)

    return (problemList, choicesList)
"""
def parseProblemAnswerFromLine(testStr):
    #input: "1038d) I will never so much as [perceive] the same air with you again ."
    #output: ['I', 'will', 'never',... ]
    pattern = "(^\d+\w\)\s)([\w|\s]+)(\s\[)(\w+)(\])([\w|\s]*)(\s\.)(\n?)$"
    #oldPattern = '(^\d+\w\s)([\w|\s]+)( \[\w+)([\w|\s]*[^\n])(\n?)$'
    m = re.match(pattern, testStr)
    problem = m.group(2) + m.group(4) if m.group(4) != '' else m.group(2)
    answer = m.group(3)

    problemWords = problem.split(' ')

    problemWordIndices = [wordToindex(word) for word in problemWords]
    answerWordIndex = wordToindex(answer[2:])
    return (problemWordIndices, answerWordIndex)
"""
# DATA_FILENAME includes TRAIN_FILE_NAME or TEST_FILE_NAME
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
            outputFile.write(str(i + 1) + ',' + ANSWERS[guessAnswer[i]] + '\n' )

def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    val = min(1, max(dotproduct(v1, v2) / (length(v1) * length(v2)), -1))
    return math.acos(val)