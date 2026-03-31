import sys
import numpy as np

filename = sys.argv[1]
numCompetitors = 0
listFile = []
dictCompetitors = {}


with open(filename, 'r') as f:

    count = 0
    for line in f:

        if count == 0:
            count += 1
            numCompetitors = int(line.strip())

        else:
            listFile.append(line.strip().split())

arrayScore = np.zeros((numCompetitors, 5), dtype = np.float32)
arrayResizeScore = np.zeros((numCompetitors, 3), dtype= np.float32)

for i in range(numCompetitors):
    combineString = listFile[i][0] + " " + listFile[i][1]
    dictCompetitors[combineString] = 0
    arrayScore[i][0] = listFile[i][3]
    arrayScore[i][1] = listFile[i][4]
    arrayScore[i][2] = listFile[i][5]
    arrayScore[i][3] = listFile[i][6]
    arrayScore[i][4] = listFile[i][7]

arrayScoreOrdered = np.sort(arrayScore, axis = 1)

for i in range(numCompetitors):
    arrayResizeScore[i] = arrayScoreOrdered[i][1:4]

scoreMartix = arrayResizeScore.sum(axis = 1)
count = 0

for comp, score in dictCompetitors.items():
    dictCompetitors[comp] = scoreMartix[count]
    count += 1

dictCompetitorsSorted = sorted(dictCompetitors.items(), key=lambda x : x[1], reverse = True)

for i in range(3):
    print(f"{i+1}: {dictCompetitorsSorted[i][0]} - Score: {dictCompetitorsSorted[i][1]}")
