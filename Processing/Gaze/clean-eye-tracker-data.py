import numpy as np
import pandas as pd
import csv
import os
import pickle


date = "08062024"
pc = "2"
convo="1"
#input file

input_file = open('../Gaze/'+date+'-Gaze-Convo'+convo+'-pc'+pc+'.txt', 'r')
# input_file = open('Gaze/Session2-Convo0-Bhasura-Gaze.txt', 'r')
# notes_file = open('3people/01-28-2022/pc3/Data/3People-1-28-2022/session/session_1. Recording 1282022 123911 PM_Notes.txt', 'r')

#output file
out_file = '../Gaze_trim/'+date+'-Gaze-Convo'+convo+'-pc'+pc+'.txt'

start = 1722971082605



stop = 1722971223738



# def read_notes():
#     global start
#     global stop
#     lines = notes_file.readlines()[1:]
#     for line in lines:
#         words = line.strip().split('\t')
#         print(words[4])
#         if words[2] == 'audio start':
#             start += int(words[4]) #use utc time

#         if words[2] == 'visual start':
#             start += int(words[4]) #use utc time

#         if words[2] == 'audio stop':
#             stop += int(words[4])

#         if words[2] == 'visual stop':
#             stop += int(words[4])

#     start = int(start/2)
#     stop = int(stop/2)
#     print("Start experiment at {0} \n".format(start))
#     print("Stop experiment at {0} \n".format(stop))

def clean_data():
    data_dict = {}
    lines = input_file.readlines()[1:]
    for line in lines: #get eye gaze data from txt file
        words = line.strip().split('\t')
        if len(words) >= 4:
            #time = words[0].split(":")
            #time[2] = time[2].replace('.','')
            #if int(time[2]) >= start and int(time[2]) <= stop:
            #    # print(time[2])
            #    # words = ['0' if x=='' else x for x in words]
            #    # print(words)
                
            #    row = [float(x) for x in words[2:]]
            #    # row.append([float(x) for x in words[2:]])
            #    data_dict[int(time[2])] = row

            #    # data.append([float(x) for x in words[1:]])

            time = words[1] #use utc
            if int(time) >= start and int(time) <= stop:
                # print(time)
                row = [float(x) for x in words[2:]]
                data_dict[int(time)] = row
                
    keys_list = list(data_dict.keys()) #utc time for each processed frame

    #for i in range(len(keys_list)-1):
    i = 0
    while i < len(keys_list)-1:
        if keys_list[i+1] < keys_list[i] + 15: 
            # print(keys_list[i])
            # print(keys_list[i+1])
            if data_dict[keys_list[i]] == [0.0, 0.0]:
                del data_dict[keys_list[i]]
            elif data_dict[keys_list[i+1]] == [0.0, 0.0]:
                del data_dict[keys_list[i+1]]
            i+=2
        else:
            i+=1
        #if keys_list[i+1] == keys_list[i] + 1:
        #    #print(keys_list[i])
        #    #print(keys_list[i], data_dict[keys_list[i]])
        #    #print(keys_list[i+1], data_dict[keys_list[i+1]]) 
        #    if data_dict[keys_list[i]] == [0.0, 0.0]:
        #        del data_dict[keys_list[i]]
        #    elif data_dict[keys_list[i+1]] == [0.0, 0.0]:
        #        del data_dict[keys_list[i+1]]

    print(len(data_dict))
    #with open(out_file, 'wb') as f:
    #    for key, value in data_dict.items():
    #        np.savetxt(f, [value], delimiter=",", newline="\n")

    ### save data to new file
    f = csv.writer(open(out_file,"w", newline=''))
    for key,value in data_dict.items():
        f.writerow([key, float(value[0]), float(value[1])])



if __name__ == '__main__':
    # read_notes()
    clean_data()
