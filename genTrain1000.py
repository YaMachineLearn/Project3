dataWordIndices = []
DATA_FILENAME = "data/training_v5_noTag_byChoice_noWordsWithMarks_3584_1737297.txt"
with open(DATA_FILENAME) as trainFeatFile:
    for line in trainFeatFile:
        dataWordIndices.append(line)

LOAD_FILENAME = "trainImportant.txt"
with open(LOAD_FILENAME) as file:
    line = file.readline()
    rowList = line.rstrip().split(" ")
    #newTrainWordIndices = [trainWordIndices[index] for index in rowList]

    with open("train_10000.txt", 'w') as outputFile:
        for index in rowList:
            outputFile.write(dataWordIndices[int(index)])
