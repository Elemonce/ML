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
        predictions = [self._predict_one_point(x) for x in X]
        return predictions


    def _predict_one_point(self, x):
        # Compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]


        # Get the closest k's
        # np.argsort() return the indices that would sort an array
        # So np.argsort([4, 2, 3, 1]) would return [3, 1, 2, 0]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        
        # Majority vote
        # return only the first tuple of the list, access the first element of that list (a tuple), access the first element of that tuple
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]

        return most_common



