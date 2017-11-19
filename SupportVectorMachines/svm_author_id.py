#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#Reduce the size of the data to 1% of its original size 
features_train = features_train[:len(features_train)]
labels_train = labels_train[:len(labels_train)]




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C = 10000.0)

t0 = time() 
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
arr = clf.predict(features_test)

count = 0
for i in arr:
    if i == 1:
        count += 1
print(count)

print clf.score(features_test, labels_test)
print("accuracy time:", round(time()-t0, 3), "s")

#########################################################


# c = 10.0  Accuracy -----> 0.61
## c = 100.0  Accuracy -----> 0.61
# c = 1000.0  Accuracy -----> 0.82
# c = 10000o.0  Accuracy -----> 0.89
