filename = "data/ex3_data.txt"

dictionary_city = {}
dictionary_month = {'Jennuary': 0, "Febrary":0, "March": 0, "April":0, "May": 0, "June": 0, "July": 0, "August": 0, "September": 0, "October": 0, "November": 0, "December":0}
birthCounter = 0
cityCounter = 0

with open(filename, 'r') as f:
    start_list= []

    for line in f:
        start_list.append(line.strip().split())

for val in start_list:
    birthCounter += 1
    if val[2] not in dictionary_city:
        dictionary_city[val[2]] = 1
    else:
        dictionary_city[val[2]] += 1

    date = val[3].split("/")[1]

    if date == "01":
        dictionary_month["Jennuary"] += 1
    if date == "02":
        dictionary_month["Febrary"] +=1
    if date == "03":
        dictionary_month["March"] += 1
    if date == "04":
        dictionary_month["April"] +=1
    if date == "05":
        dictionary_month["May"] += 1
    if date == "06":
        dictionary_month["June"] +=1
    if date == "07":
        dictionary_month["July"] += 1
    if date == "08":
        dictionary_month["August"] +=1
    if date == "09":
        dictionary_month["September"] += 1
    if date == "10":
        dictionary_month["October"] +=1
    if date == "11":
        dictionary_month["November"] += 1
    if date == "12":
        dictionary_month["December"] +=1    


print("Births per city:")

for city, count in dictionary_city.items():
    cityCounter += 1
    print(f"{city}: {count}")

print("Births per month:")

for month, count in dictionary_month.items():
    if count != 0:
        print(f"{month}: {count}")

print("Average number of births: "+ str(birthCounter/cityCounter))