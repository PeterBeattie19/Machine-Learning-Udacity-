
""" Description of Program: given a set of n points and n labels (each label corresponds to each point) for example
    given the point (1,2) and label 2, we say that point (1,2) is classified as being of type 2. 
    The end goal of this program is to train a classifier and give it point without a label,
    the program should then predict/guess the label of that point"""

import numpy as np 
X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])  #Points plotted on the graph, given in the form of [x,y] 
Y = np.array([1,1,1,2,2,2])                                #each point has a label, either a point has the property 1 or 2  
                                                           #each label's index position corresponds to the point in the array obove, with the same index 
from sklearn.naive_bayes import GaussianNB  
clf = GaussianNB()                          #construct new GaussianNB object
clf.fit(X,Y)                                #train the classifier 
GaussianNB()

print(clf.predict([[2,1]]))                 #ask the classifier to predict the label for a given point. 

