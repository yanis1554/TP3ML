#%% PART 1: Basic steps

from sklearn import tree
from matplotlib import pyplot as plt  

# Classification binaire

# X is the training set 
# Each example in X has 4 binary features
X = [[0, 0, 1, 0], [0, 1, 0, 1] , [1, 1, 0, 0] , [1, 0, 1, 1] , [0, 0, 0, 1] , [1, 1, 1, 0]]

# Y is the classes associated with the training set. 
# For instance the label of the first and second example is 1; of the third example is 0, etc
Y = [1, 1, 0, 0, 1, 1]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)


# Prediction sur un cas quelconque

clf.predict([[1,1,1,1] , [0,1,0,0] , [1,1,0,1] ])

# Resultat: array([0, 1, 0])



#%% PART 2 : Visualization

print("Prediction",clf.predict([[0,0,1,1] , [1,1,1,1] , [0,1,1,1] ]))

# Arbre de décision première version

text_representation = tree.export_text(clf)
print(text_representation)

# Arbre de décision version 2

fig = plt.figure(figsize=(10,7))
_ = tree.plot_tree(clf, 
                   feature_names= ("Big_Size","Carnivore" , "Reproduction", "Solitary"),
                   class_names= ("Not_Extinct", "Extinct" ), 
                   filled=True)


#%% PART 3: Base de données COMPASS


import csv
import numpy as np
from utils import load_from_csv

train_examples, train_labels, features, prediction = load_from_csv("./compass.csv")

# Build severals decision trees (different parameters) and visualize them

# Decision tree 1

clf_1 = tree.DecisionTreeClassifier(splitter='best',max_depth=(6),min_samples_leaf=1)
clf_1 = clf_1.fit(X, Y)

fig_1 = plt.figure(figsize=(10,7))
_ = tree.plot_tree(clf_1, 
                   feature_names= ("Big_Size","Carnivore" , "Reproduction", "Solitary"),
                   class_names= ("Not_Extinct", "Extinct" ), 
                   filled=True)

# Decision tree 2

clf_2 = tree.DecisionTreeClassifier(splitter='best',max_depth=(6),min_samples_leaf=30)
clf_2 = clf_2.fit(X, Y)

fig_2 = plt.figure(figsize=(10,7))
_ = tree.plot_tree(clf_2, 
                   feature_names= ("Big_Size","Carnivore" , "Reproduction", "Solitary"),
                   class_names= ("Not_Extinct", "Extinct" ), 
                   filled=True)


# Decision tree 3

clf_3 = tree.DecisionTreeClassifier(splitter='best',max_depth=(6),min_samples_leaf=500)
clf_3 = clf_3.fit(X, Y)

fig_3 = plt.figure(figsize=(10,7))
_ = tree.plot_tree(clf_3, 
                   feature_names= ("Big_Size","Carnivore" , "Reproduction", "Solitary"),
                   class_names= ("Not_Extinct", "Extinct" ), 
                   filled=True)
