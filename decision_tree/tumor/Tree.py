import pandas as pd
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold 
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_examples_split=2, max_depth=100, n_features=None):
        self.min_examples_split = min_examples_split
        self.max_depth = max_depth
        # This is to only pick a number of features to train instead of all of the features
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_examples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_examples < self.min_examples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # total number of features that we have; number of features that we want to select from our object; false makes it select only unique 
        # features
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        # print(f"{n_features}, {self.n_features}, {feat_idxs}, {len(feat_idxs)}")
        # print(self.n_features)
        # print(feat_idxs)
        
        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            # print(feat_idx)
            # print(X[:, 3])
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self.information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    
    def information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 

        # calculate the weighted avg. entropy of children
        n = len(y)

            # number of examples
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
            # nl_n / n, nr_n / n are weighted averages; it means how many examples are in one and in the other vector
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        # print(X_column)
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        # print(left_idxs)
        right_dxs = np.argwhere(X_column > split_thresh).flatten()

        return left_idxs, right_dxs
    
    def _entropy(self, y):
        # count the number of occurences of each number starting from 0 up until the last int in list incldued
        hist = np.bincount(y)
        ps = hist / len(y)
        # print(hist, y, ps)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])



    def _most_common_label(self, y):
        counter = Counter(y)
        # Get the most common label; get the most common tuple; first information that includes the value
        # value = counter.most_common(1)[0]
        value = counter.most_common(1)[0][0]
        # print(counter.most_common(1))
        # print(counter.most_common(1)[0])
        # print(counter.most_common(1)[0][0])
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        # print(f"Node feature: {x[node.feature]}, Node threshold: {node.threshold}")
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
