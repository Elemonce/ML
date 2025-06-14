from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from Tree2 import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state=1234
)

# classifier
clf = DecisionTree(max_depth=2)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    # y_test == y_pred returns either True or False which can also be treated as 1 or 0
    return np.sum(y_test == y_pred) / len(y_test)

# print(predictions)

print(accuracy(y_test, predictions))