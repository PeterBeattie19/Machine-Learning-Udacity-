from sklearn import tree
features_train, labels_train, features_test, labels_test = getData()

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features_train, labels_train)

accuracy = clf.score(features_test, labels_test)

print(accuracy)
