from os import walk
import os
import csv

dict = {}
for (dirpath, dirnames, filenames) in walk("."):
    for audiofile in filenames:
        if audiofile.endswith(".wav"):
            key = audiofile.split("_")[0]
            if not dict.has_key(key):
                dict[key] = []
            dict[key].append(audiofile)
            #dict[key].append("file \'" + audiofile + "\'")
    break

keys = dict.keys()
for key in keys:
    audioName = key + "_merged.wav"
    print "merging audios to " + audioName
    command = "sox "
    for value in dict[key]:
        command = command + value + " "
    command = command + audioName
    os.system(command)
    

#for key in keys:
#    txtName = key + ".txt"
#    txtfile = open(txtName, "w")
#    for value in dict[key]:
#        txtfile.write(value + "\n")
        
#for key in keys:
#    txtName = key + ".txt"
#    audioName = key + ".wav"
#    os.system('ffmpeg -f concat -i ' + txtName + ' -c copy ' + audioName) 