import csv
import numpy as np

COLUMN_HEADER = "participant_id"
PARTITIONS = ["dev", "train"]
PARTITION_EXTENSION = "_split.csv"
ID_EXTENSION = "_merged.wav_st.csv"
DIR_SOURCE = "extractedFeatures/"

def retrieveIdsFromCsv(filename):
	ids = []
	bins = []
	scores = []
	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			token = row[0]
			if token.lower() == COLUMN_HEADER:
				continue
			id = token + ID_EXTENSION
			bin = int(row[1])
			score = int(row[2])
			
			ids.append(id)
			bins.append(bin)
			scores.append(score)
	return ids, bins, scores

def averageFeatureInFile(csvFilename):
	if csvFilename.endswith(".csv"):
		with open(csvFilename, 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			tempFeatureRows = []
			for row in spamreader:
				tempFeatureRow = []
				for cell in row:
					tempFeatureCellValue = float(cell)
					tempFeatureRow.append(tempFeatureCellValue)
				tempFeatureRow = np.array(tempFeatureRow)
				tempFeatureRows.append(tempFeatureRow)
			tempFeatureRows = np.array(tempFeatureRows)
			averageRow = np.mean(tempFeatureRows, axis=0)
			stdRow = np.std(tempFeatureRows, axis=0)
			featureRow = np.concatenate([averageRow, stdRow])

			return featureRow
			
def writeFeatureToFile(filename, rows):
	f = open(filename, 'w')
	for row in rows:
		for i in range(len(row)):
			value = str(row[i])
			if i > 0:
				f.write(', ')
			f.write(value)
		f.write('\n')
		
def writeToFile(filename, rows):
	f = open(filename, 'w')
	for row in rows:
		row = str(row)
		f.write(row)
		f.write('\n')
	
def main():
	for partition in PARTITIONS:
		ids, bins, scores = retrieveIdsFromCsv(partition + PARTITION_EXTENSION)
		featureRows = []
		for id in ids:
			id_file = DIR_SOURCE + partition + "/" + id	
			featureRow = averageFeatureInFile(id_file)
			featureRows.append(featureRow)
		writeFeatureToFile(partition + "X.txt", featureRows)	
		writeToFile(partition + "Y_bin.txt", bins)
		writeToFile(partition + "Y_sev.txt", scores)

if __name__ == "__main__": 
	main()