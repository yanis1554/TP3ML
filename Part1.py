import matplotlib.pyplot as plt
import numpy as np 
import csv

from sklearn import tree
from utils import load_from_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier


# X is the training set 
# Each example in X has 4 binary features
X = [[0, 0, 1, 0], [0, 1, 0, 1] ,  [1, 1, 0, 0] , [1, 0, 1, 1] , [0, 0, 0, 1] , [1, 1, 1, 0]]
# Y is the classes associated with the training set. 
# For instance the label of the first and second example is 1; of the third example is 0, etc
Y = [1, 1, 0, 0, 1, 1]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

predicted_tree = clf.predict([[1,1,1,1] , [0,1,0,0] , [1,1,0,1] ])
