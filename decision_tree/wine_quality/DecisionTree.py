import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=5, min_features_split=2, value=None):
        self.max_depth = max_depth
        self.min_features_split = min_features_split
        self.value = value

    def _grow_tree(self):
        pass

    def fit(self, X, y):
        pass

    def _best_split(self):
        pass

    def _most_common_element(self): 
        pass

    def predict(self):
        pass