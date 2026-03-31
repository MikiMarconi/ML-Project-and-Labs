import sys 

filename = sys.argv[1]
l = []
f = open(filename, 'r')

for line in f:
    line.strip()
    l.append(line)
f.close()

finallist = []

for i in l:
    lista = i.split()
    floatscore=[]
    for x in lista[3:]:
        floatscore.append(float(x))
    
    floatscore.sort()
    floatscore = floatscore[1:4]
    sum=0
    for x in floatscore:
        sum+=x
    lista = lista[0:3]
    lista.append(sum)
    finallist.append(lista)

finallist.sort(key = lambda x: x[3], reverse = True)

print("final ranking: ")
for i in range(3):
    print(str(i+1)+":"+ finallist[i][0]+ " " + finallist[i][1]+ " - Score: "+ str(finallist[i][3]))

dictionary = {}
for i in finallist:
    if i[2] not in dictionary:
        dictionary[i[2]] = i[3]
    else:
        dictionary[i[2]]= dictionary[i[2]] + i[3]

ordered_dict = dict(sorted(dictionary.items(), key=lambda item:item[1], reverse=True))

print("Best Country: \n" + list(ordered_dict.keys())[0] + " - Total score: " + str(list(ordered_dict.values())[0]))
