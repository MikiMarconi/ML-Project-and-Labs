import sys

filename = sys.argv[1]
flag = sys.argv[2]
line_list= []
dist_list = []
dictionary = {}
total_dist = 0
total_time = 0

f = open(filename, 'r')

for line in f:
    line_list.append(line.strip().split())
f.close()

line_list.sort(key = lambda x: x[4])

for val in line_list:

    if flag == '-b':
        busid = sys.argv[3]
        if val[0] == busid:
            dist_list.append(val)

    elif flag == '-l':
        lineid = sys.argv[3]
        if val[1] == lineid:
            dist_list.append(val)

if flag == '-b':
    for i in range(len(dist_list)-1):
        dist = (((int(dist_list[i][2])-int(dist_list[i+1][2]))**2) + ((int(dist_list[i][3])-int(dist_list[i+1][3]))**2))**0.5
        total_dist += dist
    print(busid+ " - Total Distance: "+ str(total_dist))

if flag == '-l':
    for record in dist_list:
        point_list = [record[2], record[3], record[4]]
        if record[0] not in dictionary:
            dictionary[record[0]] = [point_list]
        else:
            dictionary[record[0]].append(point_list)

    
    for bus_id, list_point in dictionary.items():
        
        total_time += int(list_point[len(list_point)-1][2]) - int(list_point[0][2]) 

        for j in range(len(list_point)-1):
            dist = (((int(list_point[j][0])-int(list_point[j+1][0]))**2) + ((int(list_point[j][1])-int(list_point[j+1][1]))**2))**0.5
            total_dist += dist

    print(lineid+ " - Avg Speed: "+ str(total_dist/total_time)) 




