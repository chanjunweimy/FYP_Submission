import numpy as np
from sklearn import preprocessing

import csv

from classifierWithFS import getClassifiers, getAllFeatures, getMeanFeatures, preLoadData, selectFeaturesWithFeatureStandardization, testPerformance, getTestFeatures, getClassifiersWithoutFS


def automaticChooseFeatures(hasFs, maxFeature, featureChoices, i, X_train, X_dev, y_bin_train, y_bin_dev):
	chosenFeatureAndModels = []  
	if i == len(featureChoices):
		X_train2, X_dev2, chosenFeatures, numOfFeatures = selectFeaturesWithFeatureStandardization(maxFeature, featureChoices, X_train, X_dev)
		
		classifiers = None
		if hasFs:
			classifiers = getClassifiers()
		else:
			classifiers = getClassifiersWithoutFS()
		
		if numOfFeatures == 0:
			return chosenFeatureAndModels
		
		models_f1, models_performances = testPerformance(classifiers, X_train2, y_bin_train, X_dev2, y_bin_dev)
		chosenFeatureAndModel = (chosenFeatures, numOfFeatures, models_f1[len(models_f1)-1])
		chosenFeatureAndModels.append(chosenFeatureAndModel)
	elif i < len(featureChoices):
		#not choose
		featureChoices[i] = (featureChoices[i][0], False, featureChoices[i][2], featureChoices[i][3])
		chosenFeatureAndModels = automaticChooseFeatures(hasFs, maxFeature, featureChoices, i+1, X_train, X_dev, y_bin_train, y_bin_dev)
		
		#choose
		featureChoices[i] = (featureChoices[i][0], True, featureChoices[i][2], featureChoices[i][3])
		chosenFeatureAndModels = chosenFeatureAndModels + automaticChooseFeatures(hasFs, maxFeature, featureChoices, i+1, X_train, X_dev, y_bin_train, y_bin_dev)
	return chosenFeatureAndModels



def saveAsTable(filename, chosenFeatureAndModels):	
	with open(filename, 'wb') as csvfile:
		fieldnames = ['Index', 'Feature Used', 'Num of Audio Features', 'Num of Features', 'Best F1 mean', 'Best F1 score', 'Best Machine Learning Model']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		index = 1
		writer.writeheader()
		for chosenFeature, numOfFeatures, model_f1 in chosenFeatureAndModels:
			writer.writerow({'Index':str(index), 'Feature Used':','.join(chosenFeature), 'Num of Audio Features':len(chosenFeature), 'Num of Features':numOfFeatures, 'Best F1 mean':str(model_f1[1]), 'Best F1 score':str(model_f1[2]) + '(' + str(model_f1[3]) + ')', 'Best Machine Learning Model':model_f1[0]})
			index = index + 1

def main():
	#maxFeature, featureChoices = getAllFeatures()
	
	#get only mean
	maxFeature, featureChoices = getMeanFeatures()
	
	#use small feature sets to test
	#maxFeature, featureChoices = getTestFeatures()

	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()
	
	chosenFeatureAndModels = automaticChooseFeatures(False, maxFeature, featureChoices, 0, X_train, X_dev, y_bin_train, y_bin_dev)
	saveAsTable('classifierProposedWithFeature.csv', chosenFeatureAndModels)
		
if __name__ == "__main__": 
	main()
