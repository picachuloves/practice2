


###
print('SVM')
clf = svm.SVC()
clf.fit(x_train, y_train)
svm_predict = clf.predict(x_test)

print(cross_val_score(clf, x_train, y_train, cv=5))
print(accuracy_score(y_expect, svm_predict))
print(recall_score(y_expect, svm_predict))
print(precision_score(y_expect, svm_predict))
print(f1_score(y_expect, svm_predict))

print('SVM Scaled')

clfScaled = svm.SVC()
clfScaled.fit(scaled_x_train_array, y_train)
svm_predict = clfScaled.predict(scaled_x_test_array)

print(cross_val_score(clfScaled, scaled_x_train_array, y_train, cv=5))
print(accuracy_score(y_expect, svm_predict))
print(recall_score(y_expect, svm_predict))
print(precision_score(y_expect, svm_predict))
print(f1_score(y_expect, svm_predict))

###
print('Des Tree')
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
svm_predict = clf.predict(x_test)

print(cross_val_score(clf, x_train, y_train, cv=5))
print(accuracy_score(y_expect, svm_predict))
print(recall_score(y_expect, svm_predict))
print(precision_score(y_expect, svm_predict))
print(f1_score(y_expect, svm_predict))

print('Des Tree Scaled')

clfScaled = tree.DecisionTreeClassifier()
clfScaled.fit(scaled_x_train_array, y_train)
svm_predict = clfScaled.predict(scaled_x_test_array)

print(cross_val_score(clfScaled, scaled_x_train_array, y_train, cv=5))
print(accuracy_score(y_expect, svm_predict))
print(recall_score(y_expect, svm_predict))
print(precision_score(y_expect, svm_predict))
print(f1_score(y_expect, svm_predict))


###



