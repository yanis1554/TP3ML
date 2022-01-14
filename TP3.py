#%% PART 1: Basic steps

from sklearn import tree
from matplotlib import pyplot as plt  

# binary classification

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

# Where:

#     figsize restrains the size of the plot,
#     feature_names gives the names of the different features,
#     class_names corresponds to human readable labels for each class,
#     filled is a boolean indicating a preference to show a colorful tree.
