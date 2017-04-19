from os import walk
import os
import csv

for (dirpath, dirnames, filenames) in walk("."):
    for audiofile in filenames:
        if audiofile.endswith(".wav"):
            #print audiofile
            transcript = audiofile.replace("AUDIO_clean.wav", "TRANSCRIPT.csv")
            #print transcript
            with open('../transcript/' + transcript, 'rb') as csvfile:
                spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
                i = 0
                for row in spamreader:
                    if not row:
                        continue
                    elif row[0] == 'start_time':
                        continue
                    print row[0]
                    newAudioName = audiofile.replace("clean.wav", str(i) + ".wav")
                    i = i + 1
                    length = float(row[1]) - float(row[0])
                    min = int(float(row[0])) / 60
                    sec = float(row[0]) - min * 60.0
                    minStr = str(min)
                    secStr = str(sec)
                    if min < 10:
                        minStr = '0' + minStr
                    if int(sec) < 10:
                        secStr = '0' + secStr
                    startStr = '00:' + minStr + ':' + secStr
                    
                    lengthStr = '00:00:' + str(length)
                    if row[2] == 'Participant':
                        os.system('ffmpeg -i ' + audiofile + ' -vn -ss ' + startStr + ' -t ' + lengthStr + ' ' + newAudioName)
    break
    
    
#for (dirpath, dirnames, filenames) in walk("../transcript"):
#    print dirpath
#    print dirnames
#    print filenames
#    break