print("Loading")
from typing import NewType
from numpy.core.numeric import indices
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

data = pd.read_csv('C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Assignment2/heart.csv')
#C:\Users\jdbai\Documents\GraduateSchool\Term3\ECE9063\Assignment2
#Separate Data
x = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=34)
y_test.to_csv('C:/Users/jdbai/Desktop/y_test_svm.csv')
data2 = pd.read_csv('C:/Users/jdbai/Desktop/y_test_svm.csv', index_col=False)
new_test = data2['HeartDisease']

nn = MLPClassifier()

# parameter_space_test11 = {
#     'hidden_layer_sizes': [(11, 11, 11, 11,11, ), (11, 11, 11, 11, ), (11, 11, 11, ), (11, 11, ), (11, )],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd'],
#     'alpha': [0.0001],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [800]
# }
# parameter_space_test66 = {
#     'hidden_layer_sizes': [(66, 66, 66, 66, 66, ), (66, 66, 66, 66, ), (66, 66, 66, ), (66, 66, ), (66, )],
#     'activation': ['tan   h', 'relu'],
#     'solver': ['sgd'],
#     'alpha': [0.0001],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [800]
# }

# parameter_space_test121 = {
#     'hidden_layer_sizes': [(121, 121, 121, 121, 121, ), (121, 121, 121, 121, ), (121, 121, 121, ), (121, 121, ), (121, )],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd'],
#     'alpha': [0.0001],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [800]
# }
# parameter_space_test176 = {
#     'hidden_layer_sizes': [(176, 176, 176, 176, 176, ), (176, 176, 176, 176, ), (176, 176, 176, ), (176, 176, ), (176, )],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd'],
#     'alpha': [0.0001],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [800]
# }
# parameter_space_test = {
#     'hidden_layer_sizes': [(121,121,121, ), (121, ), (66,66,), (66,)],
#     'activation': ['relu'],
#     'solver': ['sgd'],
#     'alpha': [0.0001],
#     'learning_rate': ['adaptive'],
#     'max_iter': [800]
# }

# parameter_space_sgd = {
#     'hidden_layer_sizes': [(11, 11, 11, 11, 11, ), (11, 11, 11, 11, ), (11, 11, 11, ), (11, 11, ), (11, ),
#                            (66, 66, 66, 66, 66, ), (66, 66, 66, 66, ), (66, 66, 66, ), (66, 66, ), (66,),
#                            (220, 220, 220, 220, 220,), (220, 220, 220, 220,), (220, 220, 220, ), (220,),
#                            (550, 550, 550, 550, 550,), (550, 550, 550, 550,), (550, 550, 550,), (550,),
#                            (550, 220, 66, 11, 6,), (220, 550, 220, 66, 11,), (220, 66, 11,),
#                            ],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd'],
#     'alpha': [0.0001, 0.001, 0.005, 0.01, 0.05],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [400, 800, 1000]
# }

# parameter_space_lbfg = {
#     'hidden_layer_sizes': [(11, 11, 11, 11, 11, ), (11, 11, 11, 11, ), (11, 11, 11, ), (11, 11, ), (11, ),
#                            (66, 66, 66, 66, 66, ), (66, 66, 66, 66, ), (66, 66, 66, ), (66, 66, ), (66,),
#                            (220, 220, 220, 220, 220,), (220, 220, 220, 220,), (220, 220, 220, ), (220,),
#                            (550, 550, 550, 550, 550,), (550, 550, 550, 550,), (550, 550, 550,), (550,),
#                            (550, 220, 66, 11, 6,), (220, 550, 220, 66, 11,), (220, 66, 11,),
#                            ],
#     'activation': ['tanh', 'relu'],
#     'solver': ['lbfg'],
#     'alpha': [0.0001, 0.001, 0.005, 0.01, 0.05],
#     'max_iter': [400, 800, 1000]
# }

scores = ["precision", "recall"]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(nn, parameter_space_test, scoring="%s_macro" % score)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_["mean_test_score"]
    std = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, std, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    print()
    y_pred = y_test, clf.predict(x_test)
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
