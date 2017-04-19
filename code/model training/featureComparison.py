from classifierWithFS import getClassifiersWithoutFS, getMFCCs, preLoadData, selectFeatures, selectFeaturesWithFeatureStandardization

from sklearn import preprocessing
import matplotlib.pyplot as plt


def plotMFCCsFirst2Bin(X_train, X_train_normalized):
	X_train = X_train[:,0:2]
	X_train_normalized = X_train_normalized[:,0:2]
	
	min_max_scaler = preprocessing.MinMaxScaler()
	X_train_minmax = min_max_scaler.fit_transform(X_train)
	
	X_train_warp = preprocessing.scale(X_train)
	
	plt.xlabel("MFCCs[0]")
	plt.ylabel("MFCCs[1]")
	
	plt.scatter(X_train[:,0], X_train[:,1], color='g', label='Input Scale')
	plt.scatter(X_train_normalized[:,0], X_train_normalized[:,1], color='r', label='Audio Feature Standardized')
	plt.scatter(X_train_minmax[:,0], X_train_minmax[:,1], color='b', label='Min-max scaled')
	plt.scatter(X_train_warp[:,0], X_train_warp[:,1], color='orange', label="Feature Warping")
	plt.legend(loc='upper right')	
	plt.show()

def main():
	maxFeature, featureChoices = getMFCCs()
	cls = getClassifiersWithoutFS()
	X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()
	X_train_default, X_dev_default, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
	X_train_normalized, X_dev_normalized, chosenFeatures, numOfFeatures = selectFeaturesWithFeatureStandardization(maxFeature, featureChoices, X_train, X_dev)
	plotMFCCsFirst2Bin(X_train_default, X_train_normalized)
	
if __name__ == "__main__": 
	main()