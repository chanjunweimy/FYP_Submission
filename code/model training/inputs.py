import csv
import sys
import warnings
import numpy as np
from numpy import inf

def getX(fileName):
    X = []
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        X = np.array([ [ float(eaVal) for eaVal in row] for row in reader])
        # safety to check every row
        n_feats = len(X[0])
        i = 0
        for x in X:
            i += 1
            if n_feats != len(x):
                print('Warning, some x has different number of features!!')
                print(fileName+":"+str(i)+" has "+ str(len(x)) + " features != " + str(n_feats))
                sys.exit(1)
            if np.any(np.isnan(x)):
                print("Warning:"+fileName+":"+str(i)+" has NaN values")
                sys.exit(1)
            if not np.all(np.isfinite(x)):
                print("Warning:"+fileName+":"+str(i)+" has Inf values") 
                x[np.isneginf(x)] = 0;np.finfo(np.float64).min
                x[np.isposinf(x)] = 0; np.finfo(np.float64).max
                X[i-1] = x
                #print x
                #sys.exit(1)
    return X, n_feats, len(X)

def getY(filename):
    y = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        y = [ int(row[0]) for row in reader ]
    return y, len(y)

def read_train_dev_files(trainx, devx, trainy, devy):
    warnings.filterwarnings("ignore")

    X_train, n_feats_train, k_x_train = getX(trainx)
    X_dev, n_feats_dev, k_x_dev = getX(devx)
    y_train, k_y_train = getY(trainy)
    y_dev, k_y_dev = getY(devy)

    # some sanity checks on n_feats
    if n_feats_train != n_feats_dev:
        print('Error n_feats in train and dev. They are not equal.')
        sys.exit(1)
    n_feats = n_feats_train

    # sanity checks for k_train
    if k_x_train != k_y_train:
        print('Error, train is of different size')
        sys.exit(1)
    k_train = k_x_train
    if k_x_dev != k_y_dev:
        print('Error, dev is of different size')
        sys.exit(1)
    k_dev = k_x_dev

    print("Data has " + str(n_feats) + " features and " + str(k_train) + " training points and " + str(k_dev) + " dev points." )
    return np.array(X_train), np.array(y_train), np.array(X_dev), np.array(y_dev)

    
def read_train_dev_files_with_binary(trainx, devx, trainy, devy, trainbiny, devbiny):
    warnings.filterwarnings("ignore")

    X_train, n_feats_train, k_x_train = getX(trainx)
    X_dev, n_feats_dev, k_x_dev = getX(devx)
    y_train, k_y_train = getY(trainy)
    y_dev, k_y_dev = getY(devy)
    y_bin_train, k_y_bin_train = getY(trainbiny)
    y_bin_dev, k_y_bin_dev = getY(devbiny)

    # some sanity checks on n_feats
    if n_feats_train != n_feats_dev:
        print('Error n_feats in train and dev. They are not equal.')
        sys.exit(1)
    n_feats = n_feats_train

    # sanity checks for k_train
    if k_x_train != k_y_train:
        print('Error, train is of different size')
        sys.exit(1)
    elif k_x_train != k_y_bin_train:
        print('Error, bin_train is of different size')
        sys.exit(1)
    k_train = k_x_train
    if k_x_dev != k_y_dev:
        print('Error, dev is of different size')
        sys.exit(1)
    elif k_x_dev != k_y_bin_dev:
        print('Error, bin_dev is of different size')
        sys.exit(1)
    k_dev = k_x_dev

    print("Data has " + str(n_feats) + " features and " + str(k_train) + " training points and " + str(k_dev) + " dev points." )
    return np.array(X_train), np.array(y_train), np.array(X_dev), np.array(y_dev), np.array(y_bin_train), np.array(y_bin_dev)