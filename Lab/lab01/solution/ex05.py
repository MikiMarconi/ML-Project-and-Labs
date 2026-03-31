import sys

filename = sys.argv[1]
dim = 0
listCoord = []

with open(filename, 'r') as f:
    count = 0
    for line in f:

        if count == 0:
            dim = int(line.strip())
            count += 1
        else:
            listCoord.append(line.strip().split())


listMatrix = []

for i in range(dim):
    listSupport = []
    for j in range(dim):
        listSupport.append(0)
    listMatrix.append(listSupport)

for coord in listCoord:
    x = int(coord[0])
    y = int(coord[1])

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            
            riga_luce = x + dx
            colonna_luce = y + dy
            
            if 0 <= riga_luce < dim and 0 <= colonna_luce < dim:
                distanza = max(abs(dx), abs(dy))
                
                if distanza == 0:
                    listMatrix[riga_luce][colonna_luce] += 1.0
                elif distanza == 1:
                    listMatrix[riga_luce][colonna_luce] += 0.5
                elif distanza == 2:
                    listMatrix[riga_luce][colonna_luce] += 0.2
                    
for line in listMatrix:
    print(line)