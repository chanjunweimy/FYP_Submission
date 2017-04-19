from os import walk
import os
import csv

def getY(filename):
    y = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        y = [ int(row[0]) for row in reader ]
    return y

for (dirpath, dirnames, filenames) in walk("."):
    for phq8_filename in filenames:
		print phq8_filename
        if phq8_filename.endswith("_sev.txt"):
			multiclass_filename = phq8_filename.replace("_sev.txt", "_multi.txt")
			Ys = getY(phq8_filename)
			with open(multiclass_filename, 'w') as f:
				for y in Ys:
					if y >= 0 and y <= 4:
						f.write("0\n")
					elif y >= 5 and y <= 9:
						f.write("1\n")
					elif y >= 10 and y <= 14:
						f.write("2\n")
					elif y >= 15 and y <= 19:
						f.write("3\n")
					elif y >= 20 and y <= 24:
						f.write("4\n")
	break