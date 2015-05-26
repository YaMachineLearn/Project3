import wordUtil as wu #must be imported first

import parse
import time

#TRAIN_FILE = "data/training.txt"
TEST_FILE = "data/test.txt"
#ORIGINAL_TEST_FILE = "data/testing_data.txt"
OUTPUT_FILE = "output/result.csv"



print 'Parsing problems and choices...'
t0 = time.time()
problemList, choicesList = parse.parseProblemChoices(TEST_FILE)
#choicesList: [[324,8752,4657,123,9527], ...] #choices in each problem
#problemList: [[25,896,1888], ...]  #other words in problems
t1 = time.time()
print '...costs ', t1 - t0, ' seconds'


print 'Computing Optimal Choice...'
t0 = time.time()

answerList = list()
for i in xrange(len(problemList)):
    problem = problemList[i]    #ex: [25,896,1888]
    choices = choicesList[i]    #ex: [324,8752,4657,123,9527]

    scoreOfChoices = [ sum( [wu.wordSimilarity(choice, word) for word in problem] ) for choice in choices ]
    answer = scoreOfChoices.index(max(scoreOfChoices))  # can be faster,  but this is easier to read

    answerList.append(answer)

t1 = time.time()
print '...costs ', t1 - t0, ' seconds'



print 'Writing output file...'
parse.outputCsvFileFromAnswerNumbers(answerList, OUTPUT_FILE)

# guessAnswer = []
# for i in xrange(len(answers)):
#     degSum = []
#     for j in xrange(len(answers[i])):
#         oneDegSum = 0
#         for k in xrange(len(problems[i])):
#             oneDegSum += parse.dotproduct(answers[i][j], problems[i][k])
#         degSum.append(oneDegSum)
#     guessAnswer.append(degSum.index(max(degSum)))