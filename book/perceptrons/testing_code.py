import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# df = pd.read_csv(s, header=None, encoding="utf-8")
df = pd.read_csv("iris_data.csv")

# print(df.tail())

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map 
    markers = ('o', 's', "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1    
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha = 0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f"Class {cl}",
                    edgecolor="black")
        

# select setosa and versicolor
y = df.iloc[0:100, 4].values
# iris-setosa turns to 0; all other flowers would be a 1. in this case, it's only one other flower - versicolor, hence the binary format.
y = np.where(y == "Iris-setosa", 0, 1)

# Select 0th and 2nd column; each example is a horizontal vector of two features.
X = df.iloc[0:100, [0, 2]].values
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plot_decision_regions(X, y, classifier=ppn)
