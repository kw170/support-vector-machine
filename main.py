import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

""""
print(cancer.feature_names)
print()
print(cancer.target_names)
"""

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print("SVM Accuracy:", acc)

clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print("K-Nearest Neighbors Accuracy:", acc)

""""
for i in range(len(y_pred)):
    print("Acutal:", classes[y_test[i]],"Prediction:", classes[y_pred[i]])
"""
