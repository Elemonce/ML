import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
from KNN_torch import KNNClassifier

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plt.figure()
# plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolor="k", s=20)
# plt.show()


classifier = KNN(k=5)
# classifier = KNNClassifier(k=5)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

def accuracy(y_pred, y_actual):
    return np.sum(y_pred == y_actual) / len(y_actual)


print(accuracy(predictions, y_test))