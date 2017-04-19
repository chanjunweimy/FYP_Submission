from os import walk
import os
import csv

for (dirpath, dirnames, filenames) in walk("."):
    for audiofile in filenames:
        if audiofile.endswith(".wav"):
			#print 'file: ' + audiofile
			#print 'getting noise sample.'
			
			cleanfile = audiofile.replace(".wav", "_clean.wav")
			noisefile = audiofile.replace(".wav", "_noise.wav")

			os.system("ffmpeg -i " + audiofile + " -vn -ss 00:00:00 -t 00:00:01 " + noisefile)
			
			#print 'generating noise profile.'
			os.system("sox " + noisefile + " -n noiseprof noise.prof")
			
			#print 'cleaning original audio.'
			os.system("sox " + audiofile + " " + cleanfile + " noisered noise.prof 0.21")

			#print 'cleaned audio generated.'