from math import sqrt, isnan
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn import gaussian_process as gp
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import cross_val_score#, ShuffleSplit
from inputs import read_train_dev_files_with_binary, read_train_dev_files
from plotting import plot_bar, plot_Y, plot_f1, plotDepressedAndNormalSample, plot_mean_f1, plot_Y_with_x
import numpy as np
import sys
import time
from math import sqrt,ceil
import matplotlib.mlab as mlab
import math
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from skfeature.function.statistical_based import CFS, gini_index
from skfeature.function.information_theoretical_based import CIFE, CMIM, MRMR, MIFS, ICAP, FCBF
from skfeature.function.similarity_based import reliefF, fisher_score
from skfeature.function.sparse_learning_based import RFS

import csv
  

#setup global variables
classifiers = [("KNN", None, KNeighborsClassifier(2)),
               ("Linear SVM", None, SVC(kernel="linear")),
               ("RBF SVM", None, SVC(gamma=2, C=1)),
               ("DT", None, DecisionTreeClassifier(min_samples_split=1024, max_depth=20)),
               ("RF", None, RandomForestClassifier(n_estimators=10, min_samples_split=1024,
                                                         max_depth=20)),
               ("AB", None, AdaBoostClassifier(random_state=13370)),
               #("GP ARD", ["MFCC"], gp.GaussianProcessClassifier(kernel=ard_kernel(sigma=1.2, length_scale=np.array([1]*1)))),
               ("GP-DP", None, gp.GaussianProcessClassifier(kernel=gp.kernels.DotProduct()))
               # output the confidence level and the predictive variance for the dot product (the only one that we keep in the end)
               # GP beats SVM in our experiment (qualitative advantages)
               # only keep RBF, dot product and matern on the chart
               # add a paragraph 'Processed Data'
               #1) generate the dataset with 526 features
               #2) the predictive variance and predictive mean (best and worst) of some vectors from the dot product.
               
]

classifiers_with_all = [("KNN", ["All"], KNeighborsClassifier(2)),
               ("Linear SVM", ["All"], SVC(kernel="linear")),
               ("RBF SVM", ["All"], SVC(gamma=2, C=1)),
               ("DT", ["All"], DecisionTreeClassifier(min_samples_split=1024, max_depth=20)),
               ("RF", ["All"], RandomForestClassifier(n_estimators=10, min_samples_split=1024,
                                                         max_depth=20)),
               ("AB", ["All"], AdaBoostClassifier(random_state=13370)),
               #("GP ARD", ["MFCC"], gp.GaussianProcessClassifier(kernel=ard_kernel(sigma=1.2, length_scale=np.array([1]*1)))),
               ("GP-DP", ["All"], gp.GaussianProcessClassifier(kernel=gp.kernels.DotProduct()))
               # output the confidence level and the predictive variance for the dot product (the only one that we keep in the end)
               # GP beats SVM in our experiment (qualitative advantages)
               # only keep RBF, dot product and matern on the chart
               # add a paragraph 'Processed Data'
               #1) generate the dataset with 526 features
               #2) the predictive variance and predictive mean (best and worst) of some vectors from the dot product.
               
]

classifiers_with_relief = [("KNN", ["Relief"], KNeighborsClassifier(2)),
               ("Linear SVM", ["Relief"], SVC(kernel="linear")),
               ("RBF SVM", ["Relief"], SVC(gamma=2, C=1)),
               ("DT", ["Relief"], DecisionTreeClassifier(min_samples_split=1024, max_depth=20)),
               ("RF", ["Relief"], RandomForestClassifier(n_estimators=10, min_samples_split=1024,
                                                         max_depth=20)),
               ("AB", ["Relief"], AdaBoostClassifier(random_state=13370)),
               #("GP ARD", ["MFCC"], gp.GaussianProcessClassifier(kernel=ard_kernel(sigma=1.2, length_scale=np.array([1]*1)))),
               ("GP-DP", ["Relief"], gp.GaussianProcessClassifier(kernel=gp.kernels.DotProduct()))
               # output the confidence level and the predictive variance for the dot product (the only one that we keep in the end)
               # GP beats SVM in our experiment (qualitative advantages)
               # only keep RBF, dot product and matern on the chart
               # add a paragraph 'Processed Data'
               #1) generate the dataset with 526 features
               #2) the predictive variance and predictive mean (best and worst) of some vectors from the dot product.
               
]


FILE_X_TRAIN = "data/pyAudioAnalysisNew/trainX.txt"
FILE_X_DEV = "data/pyAudioAnalysisNew/devX.txt"
FILE_Y_TRAIN = "data/pyAudioAnalysisNew/trainY_sev.txt"
FILE_Y_DEV = "data/pyAudioAnalysisNew/devY_sev.txt"
FILE_Y_TRAIN_BIN = "data/pyAudioAnalysisNew/trainY_bin.txt"
FILE_Y_DEV_BIN = "data/pyAudioAnalysisNew/devY_bin.txt"
FILE_Y_TRAIN_MULTI = "data/pyAudioAnalysisNew/trainY_multi.txt"
FILE_Y_DEV_MULTI = "data/pyAudioAnalysisNew/devY_multi.txt"

FIGURE_Y_TRAIN = "PHQ8_Train.png"
FIGURE_Y_DEV = "PHQ8_Dev.png"
FIGURE_Y_TRAIN_BIN = "Bin_Train.png"
FIGURE_Y_DEV_BIN = "Bin_Dev.png"

LABEL_Y_TRAIN = "PHQ-8 Score"
LABEL_Y_DEV = "PHQ-8 Score"
LABEL_Y_TRAIN_BIN = "Distribution of depressed (1) and normal(0) persons in train set"
LABEL_Y_DEV_BIN = "Distribution of depressed (1) and normal (0) persons in dev set"

MAX_PHQ8 = 25
MAX_BIN = 2

MAX_FEATURE_DEFAULT = 34
MAX_FEATURE_NONE = -1

def getOneVRClassifiersWithoutFS():
	oneVrClassifiers = []
	for name, modes, model in classifiers_with_all:
		element = ["OVR " + name, modes, OneVsRestClassifier(model)]
		oneVrClassifiers.append(element)
	return oneVrClassifiers
	
def getOneVOneClassifiersWithoutFS():
	oneVOneClassifiers = []
	for name, modes, model in classifiers_with_all:
		element = ["OVO " + name, modes, OneVsOneClassifier(model)]
		oneVOneClassifiers.append(element)
	return oneVOneClassifiers
		
def getClassifiers():
	return classifiers
	
def getClassifiersWithoutFS():
	return classifiers_with_all
	
def getClassifierWithRelief():
	return classifiers_with_relief

def reliefPostProc(X, y):
	n_feats = len(X[0])
	numFeatsFn = lambda n: int(ceil(sqrt(n_feats)))
	scores = reliefF.reliefF(X,y)
	indexes = range(0, len(scores))
	pairedScores = zip(scores, indexes)
	pairedScores = sorted(pairedScores, reverse=True)
	return np.array([ eaPair[1] for eaPair in pairedScores][:numFeatsFn(n_feats)])

def baselineProc(X,y):
	n_feats = len(X[0])
	return range(0,n_feats)
	
def giniProc(X,y):
	# obtain the gini_index score of each feature
	score = gini_index.gini_index(X, y)

	# rank features in descending order according to score
	idx = gini_index.feature_ranking(score)
	return idx
	
def fisherProc(X,y):
	# obtain the score of each feature on the training set
	score = fisher_score.fisher_score(X, y)

	# rank features in descending order according to score
	idx = fisher_score.feature_ranking(score)
	return idx

def convertToBitVec(featSel):
    def wrapper(X, y):
		n_feats = len(X[0])
		feats = featSel(X,y)
		bitVec = [False] * n_feats
		for eaF in feats:
			bitVec[eaF] = True
		bitVec = np.array(bitVec)
		return len(feats),bitVec
    return wrapper


# CIFE: index of selected features, F[1] is the most important feature
# CFS: index of selected features
# RELIEF: index of selected features, F[1] is the most important feature
featSelectionFns = {
	#doesn't apply feature selection
    "All": convertToBitVec(baselineProc),
	#information_theoretical_based
    "CIFE": convertToBitVec(CIFE.cife),
    "CMIM": convertToBitVec(CMIM.cmim),
	"MRMR": convertToBitVec(MRMR.mrmr),
	"MIFS": convertToBitVec(MIFS.mifs),
	"ICAP": convertToBitVec(ICAP.icap),
	"FCBF": convertToBitVec(FCBF.fcbf),
	#statistical_based
	"CFS": convertToBitVec(CFS.cfs),
    "GINI": convertToBitVec(giniProc),
	#similarity_based
	#"Relief": convertToBitVec(reliefPostProc),
    "FISHER": convertToBitVec(fisherProc),
	#"RFS": convertToBitVec(RFS.rfs)
}

def preprocessFeatSelection(X_train, y_train):
	bitVecs = {}
	for featSelName, featSel in featSelectionFns.iteritems():
		#start = time.clock()    
		numFeats,bitVec = featSel(X_train,y_train)
		#timeTaken = time.clock() - start
		bitVecs[featSelName] = bitVec
		#print(featSelName+ "," + str(numFeats) + " in "+ str(timeTaken) + "seconds")
	return bitVecs

def getClassifieresPerformancesWithFs(classifiers, models_f1, models_performances, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs): 
    for name, featSelectionMode, model in classifiers:            
		modes = featSelectionMode
		if featSelectionMode==None:
			modes = featSelectionFns.keys()
		tempF1s = []
		tempPerformances = []
		for eaMode in modes:
			#print eaMode
			f1, performance = getClassifierPerformance(model, name, eaMode, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs)
			tempF1s.append(f1)
			tempPerformances.append(performance)
			
		tempF1s=sorted(tempF1s, key=lambda l: l[1], reverse=True)
		tempPerformances=sorted(tempPerformances, key=lambda l: l[1], reverse=True)

		models_f1 = models_f1 + tempF1s
		models_performances = models_performances + tempPerformances
    return models_f1, models_performances
	
def getClassifieresPerformances(classifiers, models_f1, models_performances, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs): 
	for name, featSelectionMode, model in classifiers:            
		modes = featSelectionMode
		if featSelectionMode==None:
			modes = featSelectionFns.keys()
		tempF1s = []
		tempPerformances = []
		for eaMode in modes:
			#print eaMode
			f1, performance = getClassifierPerformance(model, name, eaMode, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs)
			tempF1s.append(f1)
			tempPerformances.append(performance)
			
		tempF1s=sorted(tempF1s, key=lambda l: l[1], reverse=True)
		tempPerformances=sorted(tempPerformances, key=lambda l: l[1], reverse=True)

		models_f1.append(tempF1s[0])
		models_performances.append(tempPerformances[0])
	return models_f1, models_performances
	
def getMultiClassifiersPerformances(classifiers, models_f1, models_performances, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs): 
	for name, featSelectionMode, model in classifiers:            
		modes = featSelectionMode
		if featSelectionMode==None:
			modes = featSelectionFns.keys()
		tempF1s = []
		tempPerformances = []
		for eaMode in modes:
			#print eaMode
			f1, performance = getMultiClassifierPerformance(model, name, eaMode, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs)
			tempF1s.append(f1)
			tempPerformances.append(performance)
			
		tempF1s=sorted(tempF1s, key=lambda l: l[1], reverse=True)
		tempPerformances=sorted(tempPerformances, key=lambda l: l[1], reverse=True)

		models_f1.append(tempF1s[0])
		models_performances.append(tempPerformances[0])
	return models_f1, models_performances
	
def getMultiClassifierPerformance(model, name, eaMode, X, Y, X_star, Y_star, bitVecs):
	bitVec = bitVecs[eaMode]

	X = X[:,bitVec]
	X_star = X_star[:,bitVec]

	model.fit(X, Y)

	y_true = Y
	y_pred = model.predict(X)
	
	f1 = f1_score(y_true, y_pred, average=None)
	mean_f1 = f1_score(y_true, y_pred, average='macro')
	accuracy = accuracy_score(y_true, y_pred)
	model_f1 = [name, mean_f1]
	performance = [name, mean_f1, f1, accuracy]
	return model_f1, performance
	
def getClassifierPerformance(model, name, eaMode, X, Y, X_star, Y_star, bitVecs):
	bitVec = bitVecs[eaMode]

	X = X[:,bitVec]
	X_star = X_star[:,bitVec]
	#print X
	
	model.fit(X, Y)
	pp_f1, pp_precision, pp_recall, pp_accuracy = classifyForF1WithY(model, X_star, Y_star, 1)
	np_f1, np_precision, np_recall, np_accuracy = classifyForF1WithY(model, X_star, Y_star, 0)
	
	mean_f1 = float(pp_f1 + np_f1) / 2.0

	if pp_f1 == 0 or np_f1 == 0:
		mean_f1 = 0
	
	f1 = [name + '(' + eaMode + ')', mean_f1, pp_f1, np_f1]
	performance = [name + '(' + eaMode + ')', mean_f1, pp_f1, pp_precision, pp_recall, pp_accuracy, np_f1, np_precision, np_recall, np_accuracy]
	return f1, performance

def classifyForF1WithY(classifier, X, Y, positive_bit):
	y_true = Y
	y_pred = classifier.predict(X)
	f1 = f1_score(y_true, y_pred, average=None)[positive_bit]
	precision = precision_score(y_true, y_pred, average=None)[positive_bit]
	recall = recall_score(y_true, y_pred, average=None)[positive_bit]
	accuracy = accuracy_score(y_true, y_pred)

	#print 'precision:' + str(precision)
	#print 'recall:' + str(recall)
	#print 'accuracy:' + str(accuracy)
	#print 'f1:' + str(f1)
	return f1, precision, recall, accuracy

def addRelatedWork(models_f1, models_performances):
    f1 = ['DepAudioNet', 0.61, 0.52, 0.70]
    performance = ['DepAudioNet', 0.61, 0.52, 0.35, 1.00, '-', 0.70, 1.00, 0.54, '-']
    models_f1.append(f1)
    models_performances.append(performance)
    return models_f1, models_performances
 
def saveNormalizationPerformance(filename, chosenFeatures, normalizationTechs, models_f1):	
	with open(filename, 'wb') as csvfile:
		fieldnames = ['Audio Feature', 'Normalization', 'Best F1 mean', 'Best F1 score', 'Best Machine Learning Model']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for audioFeature in chosenFeatures:
			for normalizationTech in normalizationTechs:
				model_f1 = models_f1[audioFeature][normalizationTech]
				writer.writerow({'Audio Feature':audioFeature, 'Normalization':normalizationTech, 'Best F1 mean':str(model_f1[1]), 'Best F1 score':str(model_f1[2]) + '(' + str(model_f1[3]) + ')', 'Best Machine Learning Model':model_f1[0]})
 
def printPerformances(models_performances):
	for performance in models_performances:
		name = performance[0]
		mean_f1 = performance[1]
		pp_f1 = performance[2]
		pp_precision = performance[3]
		pp_recall = performance[4]
		pp_accuracy = performance[5]
		np_f1 = performance[6]
		np_precision = performance[7]
		np_recall = performance[8]
		np_accuracy = performance[9]
		print name
		print ('\tF1: ' + str(pp_f1) + '(' + str(np_f1) + 
		'),Mean F1: ' + str(mean_f1) +
		' ,Precision: ' + str(pp_precision) + '(' + str(np_precision) + 
		'),Recall: ' + str(pp_recall) + '(' + str(np_recall) + 
		'),Accuracy: '  + str(pp_accuracy) + '(' + str(np_accuracy) + ')') 
		
def printSamplesStats(ys, partition):
	num0 = 0
	num1 = 0
	for y in ys:
		if y == 1:
			num1 = num1 + 1
		elif y == 0:
			num0 = num0 + 1
	print str(num0) + 'non-depressed ' + partition + ' samples and ' + str(num1) + 'depressed ' + partition + ' samples'

def testPerformanceWithFS(classifiers, X_train, y_bin_train, X_dev, y_bin_dev):
	filesNo, featuresNo = X_train.shape
	if featuresNo == 0:
		return None, None
	
	bitVecs = preprocessFeatSelection(X_train, y_bin_train)
	
	models_f1 = []
	models_performances = []
	models_f1, models_performances = getClassifieresPerformancesWithFs(classifiers, models_f1, models_performances, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs)
	#models_f1, models_performances = addRelatedWork(models_f1, models_performances)
	#models_f1=sorted(models_f1, key=lambda l: l[1])
	#models_performances=sorted(models_performances, key=lambda l: l[1])

	bitVec = bitVecs["Relief"]
	print bitVec
	
	return models_f1, models_performances

	
	
def testPerformance(classifiers, X_train, y_bin_train, X_dev, y_bin_dev):
	filesNo, featuresNo = X_train.shape
	if featuresNo == 0:
		return None, None
	
	bitVecs = preprocessFeatSelection(X_train, y_bin_train)
	
	models_f1 = []
	models_performances = []
	models_f1, models_performances = getClassifieresPerformances(classifiers, models_f1, models_performances, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs)
	#models_f1, models_performances = addRelatedWork(models_f1, models_performances)
	models_f1=sorted(models_f1, key=lambda l: l[1])
	models_performances=sorted(models_performances, key=lambda l: l[1])

	return models_f1, models_performances
	
def testMulticlassPerformance(classifiers, X_train, y_bin_train, X_dev, y_bin_dev):
	filesNo, featuresNo = X_train.shape
	if featuresNo == 0:
		return None, None
	
	bitVecs = preprocessFeatSelection(X_train, y_bin_train)
	
	models_f1 = []
	models_performances = []
	models_f1, models_performances = getMultiClassifiersPerformances(classifiers, models_f1, models_performances, X_train, y_bin_train, X_dev, y_bin_dev, bitVecs)
	#models_f1, models_performances = addRelatedWork(models_f1, models_performances)
	models_f1=sorted(models_f1, key=lambda l: l[1])
	models_performances=sorted(models_performances, key=lambda l: l[1])

	return models_f1, models_performances
	
def showPerformance(models_f1, models_performances):
	plot_f1(models_f1)
	printPerformances(models_performances)

def preLoadData():
	x_train_file_name = FILE_X_TRAIN
	x_dev_file_name = FILE_X_DEV
	y_train_file_name = FILE_Y_TRAIN
	y_dev_file_name = FILE_Y_DEV
	y_bin_train_file_name = FILE_Y_TRAIN_BIN
	y_bin_dev_file_name = FILE_Y_DEV_BIN
	if len(sys.argv) == 5:
		x_train_file_name = sys.argv[1]
		x_dev_file_name = sys.argv[2]
		y_train_file_name = sys.argv[3]
		y_dev_file_name = sys.argv[4]
	elif len(sys.argv) == 3:
		x_train_file_name = sys.argv[1]
		x_dev_file_name = sys.argv[2]
	elif len(sys.argv) == 2:    
		 x_dev_file_name = sys.argv[1]

	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = read_train_dev_files_with_binary(x_train_file_name, x_dev_file_name, y_train_file_name, y_dev_file_name, y_bin_train_file_name, y_bin_dev_file_name)
	return X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev	
	
def preLoadMulticlassData():
	x_train_file_name = FILE_X_TRAIN
	x_dev_file_name = FILE_X_DEV
	y_train_file_name = FILE_Y_TRAIN_MULTI
	y_dev_file_name = FILE_Y_DEV_MULTI
	X_train, y_train, X_dev, y_dev = read_train_dev_files(x_train_file_name, x_dev_file_name, y_train_file_name, y_dev_file_name)
	return X_train, y_train, X_dev, y_dev

def standardized(A):
	A = (A - np.mean(A)) / np.std(A)
	return A

def appendStandardizedMatrixByIndices(matrix, to_add_matrix, indices):
	temp = standardized(to_add_matrix[:,indices])
	
	if matrix == None:
		return temp
	file_num, feature_num = temp.shape
	if file_num != matrix.shape[0]:
		print "Error: File num is not the same!"
		return temp

	matrix = np.append(matrix, temp, axis=1)

	return matrix
	
def selectFeaturesWithFeatureStandardization(maxFeature, featureChoices, X_train, X_dev):
	X_train_new = None
	X_dev_new = None
	chosenFeatures = []
	numOfFeatures = 0
	
	for featureName, isSelected, startIndex, endIndex in featureChoices:
		if isSelected:
			chosenFeatures.append(featureName)
			startIndex = startIndex - 1
			endIndex = endIndex - 1
			featureLength = endIndex - startIndex
			indices = np.arange(startIndex, endIndex)
			
			X_train_new = appendStandardizedMatrixByIndices(X_train_new, X_train, indices)
			X_dev_new = appendStandardizedMatrixByIndices(X_dev_new, X_dev, indices)
			
			numOfFeatures = numOfFeatures + featureLength
			
			if maxFeature == -1:
				continue
			
			indices = indices + maxFeature
			X_train_new = appendStandardizedMatrixByIndices(X_train_new, X_train, indices)
			X_dev_new = appendStandardizedMatrixByIndices(X_dev_new, X_dev, indices)
			numOfFeatures = numOfFeatures + featureLength
			
	return X_train_new, X_dev_new, chosenFeatures, numOfFeatures
	
def selectIndividualFeatures(isNormalized, maxFeature, featureChoices, X_train, X_dev):
	X_train_ind = {}
	X_dev_ind = {}
	chosenFeatures = []
	numOfFeatures = 0
	
	for featureName, isSelected, startIndex, endIndex in featureChoices:
		if isSelected:
			goodIndices = np.array([])
			chosenFeatures.append(featureName)
			startIndex = startIndex - 1
			endIndex = endIndex - 1
			
			X_train_feature = None 
			X_dev_feature = None
			
			indices = np.arange(startIndex, endIndex)
			numOfFeatures = numOfFeatures + indices.size
			if isNormalized:
				X_train_feature = appendStandardizedMatrixByIndices(X_train_feature, X_train, indices)
				X_dev_feature = appendStandardizedMatrixByIndices(X_dev_feature, X_dev, indices)
			else:
				X_train_feature = X_train[:,indices]
				X_dev_feature = X_dev[:,indices]
			
			if maxFeature != -1:
				indices = indices + maxFeature
				numOfFeatures = numOfFeatures + indices.size
				if isNormalized:
					X_train_feature = appendStandardizedMatrixByIndices(X_train_feature, X_train, indices)
					X_dev_feature = appendStandardizedMatrixByIndices(X_dev_feature, X_dev, indices)
				else:
					X_train_feature = np.append(X_train_feature, X_train[:,indices], axis=1)
					X_dev_feature = np.append(X_dev_feature, X_dev[:,indices], axis=1)
					
			X_train_ind[featureName] = X_train_feature
			X_dev_ind[featureName] = X_dev_feature

	return X_train_ind, X_dev_ind, chosenFeatures, numOfFeatures
	
def selectFeatures(maxFeature, featureChoices, X_train, X_dev):
	goodIndices = np.array([])
	chosenFeatures = []
	
	for featureName, isSelected, startIndex, endIndex in featureChoices:
		if isSelected:
			chosenFeatures.append(featureName)
			startIndex = startIndex - 1
			endIndex = endIndex - 1
			indices = np.arange(startIndex, endIndex)
			goodIndices = np.concatenate([indices, goodIndices])
			
			if maxFeature == -1:
				continue
			
			indices = indices + maxFeature
			goodIndices = np.concatenate([indices, goodIndices])
			
	goodIndices = goodIndices.astype(int)
	numOfFeatures = goodIndices.size
	
	X_train = X_train[:,goodIndices]
	X_dev = X_dev[:,goodIndices]
	return X_train, X_dev, chosenFeatures, numOfFeatures

def getTestFeatures():
	maxFeature = MAX_FEATURE_NONE
	featureChoices = [("Spectral Rolloff", True, 8, 9),
            ("MFCCs", True, 9, 22)]
	return maxFeature, featureChoices

def getMeanFeatures():
	maxFeature = MAX_FEATURE_NONE
	featureChoices = [("Zero-crossing Rate", True, 1, 2),
            ("Energy", True, 2, 3),
            ("Entropy of Energy", True, 3, 4),
            ("Spectral Centroid", True, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", True, 6, 7),
            ("Spectral Flux", True, 7, 8),
            ("Spectral Rolloff", True, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", True, 22, 34),
            ("Chroma Deviation", True, 34, 35)]
	return maxFeature, featureChoices
		
def getFeatures():
	maxFeature = MAX_FEATURE_DEFAULT
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", False, 3, 4),
            ("Spectral Centroid", False, 4, 5),
            ("Spectral Spread", False, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", False, 34, 35)]
	return maxFeature, featureChoices

def getMeanMFCCs():
	maxFeature = MAX_FEATURE_NONE
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", False, 3, 4),
            ("Spectral Centroid", False, 4, 5),
            ("Spectral Spread", False, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", False, 34, 35)]
	return maxFeature, featureChoices		
	
def getMFCCs():
	maxFeature = MAX_FEATURE_DEFAULT
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", False, 3, 4),
            ("Spectral Centroid", False, 4, 5),
            ("Spectral Spread", False, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", False, 34, 35)]
	return maxFeature, featureChoices	
	
def getAllFeatures():
	maxFeature = MAX_FEATURE_DEFAULT
	featureChoices = [("Zero-crossing Rate", True, 1, 2),
            ("Energy", True, 2, 3),
            ("Entropy of Energy", True, 3, 4),
            ("Spectral Centroid", True, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", True, 6, 7),
            ("Spectral Flux", True, 7, 8),
            ("Spectral Rolloff", True, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", True, 22, 34),
            ("Chroma Deviation", True, 34, 35)]
	return maxFeature, featureChoices
	
def getBestMeanFeatures():
	maxFeature = MAX_FEATURE_NONE
	featureChoices = [("Zero-crossing Rate", True, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", True, 3, 4),
            ("Spectral Centroid", False, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", True, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", True, 34, 35)]
	return maxFeature, featureChoices
	
def getBestNormalizedMeanFeatures():
	maxFeature = MAX_FEATURE_NONE
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", False, 3, 4),
            ("Spectral Centroid", True, 4, 5),
            ("Spectral Spread", False, 5, 6),
            ("Spectral Entropy", True, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", True, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", True, 22, 34),
            ("Chroma Deviation", True, 34, 35)]
	return maxFeature, featureChoices	
	
def getBestMeanStdFeatures():
	maxFeature = MAX_FEATURE_DEFAULT
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", False, 3, 4),
            ("Spectral Centroid", True, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", True, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", False, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", False, 34, 35)]
	return maxFeature, featureChoices
	
def getBestNormalizedMeanStdFeatures():
	maxFeature = MAX_FEATURE_DEFAULT
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", True, 3, 4),
            ("Spectral Centroid", True, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", True, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", False, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", False, 34, 35)]
	return maxFeature, featureChoices

def plotDistribution():	
	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()

	plot_Y(y_train, y_bin_train, FIGURE_Y_TRAIN, LABEL_Y_TRAIN, MAX_PHQ8)
	plot_Y(y_dev, y_bin_dev, FIGURE_Y_DEV, LABEL_Y_DEV, MAX_PHQ8)
	plotDepressedAndNormalSample()

def testEveryFeatureSelectionMethods():
	maxFeature, featureChoices = getAllFeatures()
	cls = getClassifiers()
	
	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()

	printSamplesStats(y_bin_train, "training")
	printSamplesStats(y_bin_dev, "dev")
	
	X_train_default, X_dev_default, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
	
	models_f1, models_performances = testPerformanceWithFS(cls, X_train_default, y_bin_train, X_dev_default, y_bin_dev)
	showPerformance(models_f1, models_performances)
	
def testRelief():
	maxFeature = MAX_FEATURE_DEFAULT
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", True, 3, 4),
            ("Spectral Centroid", True, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", True, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", False, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", False, 34, 35)]
	#cls = [("AB", ["Relief"], AdaBoostClassifier(random_state=13370))]
	
	cls = getClassifierWithRelief()
	
	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()

	printSamplesStats(y_bin_train, "training")
	printSamplesStats(y_bin_dev, "dev")
	
	X_train_default, X_dev_default, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
	
	models_f1, models_performances = testPerformanceWithFS(cls, X_train_default, y_bin_train, X_dev_default, y_bin_dev)
	showPerformance(models_f1, models_performances)
	
	
def testEveryFeatureWithoutFS():
	normalizationTechs = ["None", "Audio Feature Standardization"]
	filename = "normalization_comparison.csv"

	maxFeature, featureChoices = getAllFeatures()
	#cls = [("AB", ["All"], AdaBoostClassifier(random_state=13370))]
	cls = getClassifiersWithoutFS()
	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()
	
	#not normalized
	X_train_ind, X_dev_ind, chosenFeatures, numOfFeatures = selectIndividualFeatures(False, maxFeature, featureChoices, X_train, X_dev)
	
	#normalized
	X_train_proposed_ind, X_dev_proposed_ind, chosenFeatures, numOfFeatures = selectIndividualFeatures(True, maxFeature, featureChoices, X_train, X_dev)

	featureDict = {}
	
	for audioFeature in chosenFeatures:
		normalizedDict = {}
		#not normalized
		X_train_normal = X_train_ind[audioFeature]
		X_dev_normal = X_dev_ind[audioFeature]
		models_f1, models_performances = testPerformance(cls, X_train_normal, y_bin_train, X_dev_normal, y_bin_dev)
		#print models_performances
		normalizedDict[normalizationTechs[0]] = models_f1[len(models_f1)-1] 

		#normalized
		X_train_normalized = X_train_proposed_ind[audioFeature]
		X_dev_normalized = X_dev_proposed_ind[audioFeature]
		models_f1, models_performances = testPerformance(cls, X_train_normalized, y_bin_train, X_dev_normalized, y_bin_dev)
		normalizedDict[normalizationTechs[1]] = models_f1[len(models_f1)-1]
		
		featureDict[audioFeature] = normalizedDict
	
	saveNormalizationPerformance(filename, chosenFeatures, normalizationTechs, featureDict)
	
def combineMeanAndMeanStd():
	maxFeature = MAX_FEATURE_DEFAULT
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", False, 3, 4),
            ("Spectral Centroid", False, 4, 5),
            ("Spectral Spread", False, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", False, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", False, 34, 35)]
	
	meanMaxFeature = MAX_FEATURE_NONE
	meanFeatureChoices = [("Zero-crossing Rate", True, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", True, 3, 4),
            ("Spectral Centroid", False, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", True, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", True, 34, 35)]
			
	cls = getClassifiers()

	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()
	
	X_train_default, X_dev_default, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
	X_train_mean, X_dev_mean, chosenFeatures, numOfFeatures = selectFeatures(meanMaxFeature, meanFeatureChoices, X_train, X_dev)
	
	X_train_proposed = np.append(X_train_default, X_train_mean, axis=1)
	X_dev_proposed = np.append(X_dev_default, X_dev_mean, axis=1)
	
	print X_train_default.shape
	print X_train_mean.shape
	print X_train_proposed.shape
	models_f1, models_performances = testPerformance(cls, X_train_proposed, y_bin_train, X_dev_proposed, y_bin_dev)
	showPerformance(models_f1, models_performances)
	
def testNormalizedMeanAudioFeatures():
	maxFeature = MAX_FEATURE_NONE
	meanFeatureChoices = [("Zero-crossing Rate", True, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", True, 3, 4),
            ("Spectral Centroid", False, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", True, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", True, 34, 35)]
			
	cls = getClassifiers()

	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()
	X_train_mean, X_dev_mean, chosenFeatures, numOfFeatures = selectFeaturesWithFeatureStandardization(maxFeature, meanFeatureChoices, X_train, X_dev)
	models_f1, models_performances = testPerformance(cls, X_train_mean, y_bin_train, X_dev_mean, y_bin_dev)
	showPerformance(models_f1, models_performances)
	
	

def combineNormalizedAndNot():
	maxFeature = MAX_FEATURE_DEFAULT
	featureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", False, 3, 4),
            ("Spectral Centroid", True, 4, 5),
            ("Spectral Spread", True, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", True, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", False, 9, 22),
            ("Chroma Vector", False, 22, 34),
            ("Chroma Deviation", False, 34, 35)]
	
	normalizedFeatureChoices = [("Zero-crossing Rate", False, 1, 2),
            ("Energy", False, 2, 3),
            ("Entropy of Energy", False, 3, 4),
            ("Spectral Centroid", False, 4, 5),
            ("Spectral Spread", False, 5, 6),
            ("Spectral Entropy", False, 6, 7),
            ("Spectral Flux", False, 7, 8),
            ("Spectral Rolloff", False, 8, 9),
            ("MFCCs", True, 9, 22),
            ("Chroma Vector", True, 22, 34),
            ("Chroma Deviation", True, 34, 35)]
	
	cls = getClassifiers()

	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()
	
	X_train_default, X_dev_default, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
	X_train_normalized, X_dev_normalized, chosenFeatures, numOfFeatures = selectFeaturesWithFeatureStandardization(maxFeature, normalizedFeatureChoices, X_train, X_dev)
	
	X_train_proposed = np.append(X_train_default, X_train_normalized, axis=1)
	X_dev_proposed = np.append(X_dev_default, X_dev_normalized, axis=1)
	
	print X_train_default.shape
	print X_train_normalized.shape
	print X_train_proposed.shape
	models_f1, models_performances = testPerformance(cls, X_train_proposed, y_bin_train, X_dev_proposed, y_bin_dev)
	showPerformance(models_f1, models_performances)

def testAdaBoostMultiClass():
	X_train, y_train, X_dev, y_dev = preLoadMulticlassData()
	
	cls = [("AB", ["All"], OneVsRestClassifier(AdaBoostClassifier(random_state=13370)))]
	
	#maxFeature, featureChoices = getAllFeatures()
	#maxFeature, featureChoices = getBestMeanFeatures()
	#maxFeature, featureChoices = getBestMeanStdFeatures()
	#maxFeature, featureChoices = getBestNormalizedMeanStdFeatures()
	#maxFeature, featureChoices = getBestNormalizedMeanFeatures()
	#maxFeature, featureChoices = getMFCCs()
	#maxFeature, featureChoices = getMeanMFCCs()
	#maxFeature, featureChoices = getMeanFeatures()
	X_train = X_train[:,0:1]
	X_train = X_train[:,0:1]
	
	X_train, X_dev, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
	#X_train, X_dev, chosenFeatures, numOfFeatures = selectFeaturesWithFeatureStandardization(maxFeature, featureChoices, X_train, X_dev)
	
	models_f1, models_performances = testMulticlassPerformance(cls, X_train, y_train, X_dev, y_dev)
	#plot_mean_f1(models_f1)	
	print models_performances
	
def plotMultiClassDistribution():
	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()
	X_train, y_train, X_dev, y_dev = preLoadMulticlassData()
	x_tickslabels = ["minimal", "mild", "moderate", "moderately severe", "severe"]
	plot_Y_with_x(y_train, y_bin_train, "DepressionLevelTrain.png", "Depression Level (Train)", 5, x_tickslabels)
	plot_Y_with_x(y_dev, y_bin_dev, "DepressionLevelDev.png", "Depression Level (Dev)", 5, x_tickslabels)
	
def multiclassClassifcation():
	X_train, y_train, X_dev, y_dev = preLoadMulticlassData()
	cls1 = getOneVRClassifiersWithoutFS()
	cls2 = getOneVOneClassifiersWithoutFS()
	cls3 = getClassifiersWithoutFS()
	#cls = cls1 + cls2
	#cls = cls + cls3
	cls = getClassifiersWithoutFS()
	
	#maxFeature, featureChoices = getBestMeanFeatures()
	#maxFeature, featureChoices = getBestMeanStdFeatures()
	#maxFeature, featureChoices = getBestNormalizedMeanStdFeatures()
	maxFeature, featureChoices = getBestNormalizedMeanFeatures()
	#X_train, X_dev, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
	X_train, X_dev, chosenFeatures, numOfFeatures = selectFeaturesWithFeatureStandardization(maxFeature, featureChoices, X_train, X_dev)
	
	models_f1, models_performances = testMulticlassPerformance(cls, X_train, y_train, X_dev, y_dev)
	plot_mean_f1(models_f1)	
	print models_performances
	
def experiment():
	#maxFeature, featureChoices = getFeatures()
	#maxFeature, featureChoices = getMeanFeatures()
	#maxFeature, featureChoices = getAllFeatures()
	maxFeature, featureChoices = getMeanMFCCs()
	#maxFeature, featureChoices = getMFCCs()
	
	#cls = getClassifiers()
	cls = getClassifiersWithoutFS()

	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()

	printSamplesStats(y_bin_train, "training")
	printSamplesStats(y_bin_dev, "dev")
	
	X_train_default, X_dev_default, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
	#print X_train_default.shape
	
	models_f1, models_performances = testPerformance(cls, X_train_default, y_bin_train, X_dev_default, y_bin_dev)
	showPerformance(models_f1, models_performances)
	
	"""
	X_train_standardized = preprocessing.scale(X_train)
	X_dev_standardized = preprocessing.scale(X_dev)
	
	models_f1, models_performances = testPerformance(cls, X_train_standardized, y_bin_train, X_dev_standardized, y_bin_dev)
	showPerformance(models_f1, models_performances)
	
	min_max_scaler = preprocessing.MinMaxScaler()
	X_train_minmax = min_max_scaler.fit_transform(X_train)
	X_dev_minmax = min_max_scaler.fit_transform(X_dev)
	models_f1, models_performances = testPerformance(cls, X_train_minmax, y_bin_train, X_dev_minmax, y_bin_dev)
	showPerformance(models_f1, models_performances)
	
	
	X_train_proposed, X_dev_proposed, chosenFeatures, numOfFeatures = selectFeaturesWithFeatureStandardization(maxFeature, featureChoices, X_train, X_dev)
	
	models_f1, models_performances = testPerformance(cls, X_train_proposed, y_bin_train, X_dev_proposed, y_bin_dev)
	showPerformance(models_f1, models_performances)
	"""
	
def main():
	#plotDistribution()
	#experiment()
	#testEveryFeatureWithoutFS()
	#testEveryFeatureSelectionMethods()
	#testRelief()
	#combineNormalizedAndNot()
	#combineMeanAndMeanStd()
	#testNormalizedMeanAudioFeatures()
	#multiclassClassifcation()
	#testAdaBoostMultiClass()
	plotMultiClassDistribution()
	
if __name__ == "__main__": 
	main()