# Author: Noel Dawe <noel.dawe@gmail.com>
# Edited b
# License: BSD 3 clause

from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from classifierWithFS import preLoadData, getBestMeanFeatures, selectFeatures


X_train, y_train_phq8, X_test, y_dev, y_train, y_test = preLoadData()
maxFeature, featureChoices = getBestMeanFeatures()

X_train, X_test, chosenFeatures, numOfFeatures = selectFeatures(maxFeature, featureChoices, X_train, X_test)

bdt_real = AdaBoostClassifier(random_state=13370)

bdt_real.fit(X_train, y_train)

real_test_errors = []

for real_test_predict in bdt_real.staged_predict(X_test):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))

n_trees_real = len(bdt_real)

# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
real_estimator_weights = bdt_real.estimator_weights_[:n_trees_real]

#plt.figure(figsize=(15, 5))

#plt.subplot(131)
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='AB')
plt.legend()
#plt.ylim(0.18, 0.62)
plt.ylabel('Dev Error')
plt.xlabel('Number of Trees')

"""
plt.subplot(132)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='AB', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
#plt.ylim((.2,
#         max(real_estimator_errors.max(),
#             discrete_estimator_errors.max()) * 1.2))
#plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)
plt.plot(range(1, n_trees_real + 1), real_estimator_weights,
         "b", label='AB')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, real_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_real + 20))

# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()
"""
plt.show()