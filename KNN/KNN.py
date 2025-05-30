import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum(x1-x2)**2)
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]

        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train_value) for x_train_value in self.X_train]

        # get the closest K
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
            # Get the tuple of the element that occurs the most for each example
            # most_common = Counter(k_nearest_labels).most_common()[0]

            # For every example, return a list of the first n most common elements.
            # However, as n is set to 1 here, it only returns the first element,
            # and the only difference between this and the previous line of code
            # is that every example is a tuple stored in a list, rather than just a tuple.
        most_common = Counter(k_nearest_labels).most_common(1)

            # Get the value of the element that occurs the most for each example
        most_common = Counter(k_nearest_labels).most_common()[0][0]

        return most_common


        