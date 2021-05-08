from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
features = iris.data
labels = iris.target

print(features[0])
print(labels[0])

classifier = KNeighborsClassifier()
classifier.fit(features, labels)

predict = classifier.predict([[1, 1, 1, 1]])
print(predict)
