from Tree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# data = datasets.load_breast_cancer()
# X, y = data.data, data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# clf = DecisionTree()
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    # y_test == y_pred return 0 or 1 and thus, we sum the 1s, which indicate the number of examples guessed correctly
    return np.sum(y_test == y_pred) / len(y_test)

# acc = accuracy(y_test, predictions)
# print(acc)

df = pd.read_csv("winequality-red.csv")

df.dropna(inplace=True)

X = df.drop(columns="quality")
y = df["quality"]

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy(y_test, predictions))