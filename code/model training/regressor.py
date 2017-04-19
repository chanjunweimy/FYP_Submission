from math import sqrt, isnan
from sklearn.svm import LinearSVC
from sklearn import gaussian_process as gp
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingRegressor
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.model_selection import cross_val_score#, ShuffleSplit
from skfeature.function.statistical_based import CFS
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.similarity_based import reliefF
from inputs import read_train_dev_files_with_binary
from plotting import plot_bar, plot_all_Y, plot_f1
import numpy as np
import sys
import GPy.kern as kern
import GPy.models as models
import time
from math import sqrt,ceil
import GPy
import GPyOpt
import matplotlib.mlab as mlab
import math
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE

from classifierWithFS import getBestNormalizedMeanFeatures, getBestMeanFeatures, getBestMeanStdFeatures, preLoadData, selectFeatures, selectFeaturesWithFeatureStandardization, getBestNormalizedMeanStdFeatures



regressors = [
               ("Linear SVR", None, SVR(kernel="linear")),
               ("RBF SVR", None, SVR(gamma=2, C=1)),
               ("DT", None, DecisionTreeRegressor(min_samples_split=1024, max_depth=20)),
               ("RF", None, RandomForestRegressor(n_estimators=10, min_samples_split=1024,
                                                         max_depth=20)),
               ("AB", None, AdaBoostRegressor(random_state=13370)),
               ("NB", None, GaussianNB()),
               ("KNN", None, KNeighborsRegressor(2)),
               #("GP isotropic RBF", None, gp.GaussianProcessRegressor(kernel=gp.kernels.RBF())),
               #("GP anisotropic RBF", ["All"], gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(length_scale=np.array([1]*n_feats)))),
               #("GP isotropic matern nu=0.5", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=0.5))),
               #("GP isotropic matern nu=1.5", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=1.5))),
               #("GP Isotropic Matern", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=2.5))),
# bad performance
               ("GP-DP", ["CFS", "CIFE", "MFCC", "All"], gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())),
               # output the confidence level and the predictive variance for the dot product (the only one that we keep in the end)
               # GP beats SVM in our experiment (qualitative advantages)
               # only keep RBF, dot product and matern on the chart
               # add a paragraph 'Processed Data'
               #1) generate the dataset with 526 features
               #2) the predictive variance and predictive mean (best and worst) of some vectors from the dot product.

#  3-th leading minor not positive definite
#    ("GP exp sine squared", gp.GaussianProcessRegressor(kernel=gp.kernels.ExpSineSquared())),
               #("GP rational quadratic", None, gp.GaussianProcessRegressor(kernel=gp.kernels.RationalQuadratic())),
               #("GP white kernel", None, gp.GaussianProcessRegressor(kernel=gp.kernels.WhiteKernel())),
               #("GP abs_exp", None, gp.GaussianProcess(corr='absolute_exponential')),
               #("GP squared_exp", ["All"], gp.GaussianProcess(corr='squared_exponential')),
               #("GP cubic", None, gp.GaussianProcess(corr='cubic')),
               #("GP linear", None, gp.GaussianProcess(corr='linear')),
               #("GP RBF ARD", ["All"], RBF_ARD_WRAPPER(kern.RBF(input_dim=n_feats, variance=1., lengthscale=np.array([1]*n_feats), ARD=True)))]
]

X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = preLoadData()

#maxFeature, featureChoices = getBestMeanFeatures()
#maxFeature, featureChoices = getBestMeanStdFeatures()
#maxFeature, featureChoices = getBestNormalizedMeanStdFeatures()
maxFeature, featureChoices = getBestNormalizedMeanFeatures()
#X_train, X_dev, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_dev)
X_train, X_dev, chosenFeatures, numOfFeatures = selectFeaturesWithFeatureStandardization(maxFeature, featureChoices, X_train, X_dev)

modeString = 'All'

models_rmse = []
for name, featSelectionMode, model in regressors:
    model.fit(X_train, y_train)
    predictTrain = model.predict(X_train)
    predictDev = model.predict(X_dev)
    rmse_train = sqrt(mean_squared_error(y_train, predictTrain))
    rmse_predict = sqrt(mean_squared_error(y_dev, predictDev))
    mae_train = mean_absolute_error(y_train, predictTrain)
    mae_dev = mean_absolute_error(y_dev, predictDev)
    models_rmse.append([name + '('+modeString+')', rmse_train, rmse_predict])
    print(name + '('+modeString+')')
    print("\trmse:\n\tT:" + str(rmse_train)+"\n\tP:"+str(rmse_predict))
    print("\tmae:\n\tT:" + str(mae_train)+"\n\tP:"+str(mae_dev))
plot_bar(models_rmse)
