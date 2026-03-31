import sys

filename = sys.argv[1]
copyDictionary = {}
dictionaryMonth = {}
dictionaryTotalPrice = {}
dictionaryCopyBought = {}
dictionaryTotalSold = {}
dictionaryCopySold = {}

with open(filename, 'r') as f:
    startList = []
    for line in f:
        startList.append(line.strip().split())

for val in startList:

    if val[1] == 'B':

        if val[0] not in copyDictionary:
            copyDictionary[val[0]] = int(val[3])
            dictionaryCopyBought[val[0]] = int(val[3])
            dictionaryTotalPrice[val[0]] = float(val[4]) * int(val[3])

        else:
            copyDictionary[val[0]] += int(val[3])
            dictionaryCopyBought[val[0]] += int(val[3])
            dictionaryTotalPrice[val[0]] += float(val[4]) * int(val[3])
        

    else:
        copyDictionary[val[0]] -= int(val[3])
        year = val[2].split("/")[2]
        month = val[2].split("/")[1]
        yearMonth = year + "/"+ month

        if yearMonth not in dictionaryMonth:
            dictionaryMonth[yearMonth] = int(val[3])

        else:
            dictionaryMonth[yearMonth] += int(val[3])


        if val[0] not in dictionaryCopySold or val[0] not in dictionaryTotalSold:
            dictionaryCopySold[val[0]] = int(val[3])
            dictionaryTotalSold[val[0]] = float(val[4]) * int(val[3])

        else:
            dictionaryCopySold[val[0]] += int(val[3])
            dictionaryTotalSold[val[0]] += float(val[4]) * int(val[3])

        

print("Available copies: ")

for isbn, nCopies in copyDictionary.items():
    print(f"{isbn}: {nCopies}")

print("Sold books per month:")

for yearMonth, nCopies in dictionaryMonth.items():
    year = yearMonth.split("/")[0]
    month = yearMonth.split("/")[1]
    if month == "01":
        print(f"Jennuary, {year}: {nCopies}")
    if month == "02":
        print(f"Febrary, {year}: {nCopies}")
    if month == "03":
        print(f"March, {year}: {nCopies}")
    if month == "04":
        print(f"April, {year}: {nCopies}")
    if month == "05":
        print(f"May, {year}: {nCopies}")
    if month == "06":
        print(f"June, {year}: {nCopies}")
    if month == "07":
        print(f"July, {year}: {nCopies}")
    if month == "08":
        print(f"August, {year}: {nCopies}")
    if month == "09":
        print(f"September, {year}: {nCopies}")
    if month == "10":
        print(f"October, {year}: {nCopies}")
    if month == "11":
        print(f"November, {year}: {nCopies}")
    if month == "12":
        print(f"December, {year}: {nCopies}")


#Costo medio
dictionaryAvgCost = {}
for isbn, cost in dictionaryTotalPrice.items():
    dictionaryAvgCost[isbn] = float(cost) / float(dictionaryCopyBought[isbn]) 

#Guadagno totale
dictionaryTotalRevenue = {}
for isbn, cost in dictionaryTotalSold.items():
    dictionaryTotalRevenue[isbn] = float(cost) - (float(dictionaryAvgCost[isbn]) * float(dictionaryCopySold[isbn]))

#Guadagno medio
dictionaryAvgRevenue = {}
for isbn, totalRev in dictionaryTotalRevenue.items():
    dictionaryAvgRevenue[isbn] = float(totalRev)/float(dictionaryCopySold[isbn])

print("Gain per book:")
for isbn, avgrev in dictionaryAvgRevenue.items():
    print(f"Total gain: {dictionaryTotalRevenue[isbn]}, Avg: {avgrev}, Sold copies: {dictionaryCopySold[isbn]}")