import sys
import numpy as np

filename = sys.argv[1]
count = 0
roomDimension = 0
coordsList = []

with open(filename, 'r') as f:
    
    for line in f:

        if count == 0:
            count += 1
            roomDimension = int(line.strip())

        else:
            coordsList.append(line.strip().split())

arrayRoom = np.zeros((roomDimension, roomDimension), dtype = np.float32)

for coords in coordsList:
    x = int(coords[0])
    y = int(coords[1])

    for dx in range(-2, 3):
        for dy in range(-2,3):
            row = x + dx
            column = y + dy
            
            if 0 <= row < roomDimension and 0 <= column < roomDimension:
                distance = max(abs(dx), abs(dy))

                if distance == 0:
                    arrayRoom[row, column] += 1
                    
                elif distance == 1:
                    arrayRoom[row, column] += 0.5

                elif distance == 2:
                    arrayRoom[row, column] += 0.2


print(arrayRoom)
