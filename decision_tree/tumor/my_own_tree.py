import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *,value=None):
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
        self.n_features = n_features

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else max(self.n_features, X.shape[1])

        self._grow_tree(X, y, self.max_depth)
    
    def _grow_tree(self, X, y, depth):
        n_examples, n_features = X.shape
        n_labels = np.unique(y)

        if n_examples < self.min_examples_split or depth >= self.max_depth or n_labels == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)


        

        best_feature, best_threshold = self._best_split(X)

        left, right = self._split(best_feature, best_threshold)

        self.left = self._grow_tree(left, y[:left.shape[0]])
        self.right = self._grow_tree(right, y[right.shape[0]:])



    def _most_common_label(self, y):
        counter = Counter(y)
        most_common_y = counter.most_common(1)[0][0]

        return most_common_y

    def _best_split(self, X, y, feat_idxs):
        best_ig = -1
        best_feature, best_thr = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                ig = self._information_gain(X_column, y, thr)
                if ig > best_ig:
                    best_ig = ig
                    best_feature = feat_idx
                    best_thr = thr


        return best_feature, best_thr

    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()

        return left_idxs, right_idxs
        

    def _information_gain(self, X_column, y, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        n_examples = len(y)
        n_left = len(left_idxs)
        n_right = len(right_idxs)

        entropy_left = self._entropy(y[left_idxs])
        entropy_right = self._entropy(y[right_idxs])

        child_entropy = (n_left / n_examples) * entropy_left + (n_right / n_examples) * entropy_right

        ig = parent_entropy - child_entropy

        return ig

    def _entropy(self, y):
        hist = np.bincount(y)

        ps = hist / len(y)

        return np.sum([p & np.log2(p) for p in ps if p > 0])

    def predict(self):
        pass


    def _traverse_tree(self):
        pass