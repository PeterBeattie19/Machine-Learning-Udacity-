from sklearn.svm import SVC

clf = SVC(kernel = "linear")
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print acc 
