import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        # Which feature this (what does "this" mean?) was divided with. I assume "this" is the branch that was divided into nodes. (Or rather this node itself).
        self.feature = feature
        # Which threshold this branch was divided with
        self.threshold = threshold
        # Left node
        self.left = left
        # Right node
        self.right = right
        # The value will only be passed to leaf nodes. Leaf nodes are nodes that determine the feature of the node. (Either having 100% purity or by the majority of votes)
        # The value of the value is the label (the prediction, e.g. Cat, Dog, Lion).
        self.value = value

    def is_leaf_node(self):
        return self.value is not None



class Decisiontree:
    def __init__(self, min_examples_split=2, max_depth=100, n_features=None):
        self.min_examples_split = min_examples_split
        self.max_depth = max_depth
        # The number of features to choose from a given pool of features. (To make it random and have different trees later)
        self.n_features = n_features
        self.root = None

    
    def fit(self, X, y):
        # if not means if it's None
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        n_examples, n_features = X.shape
        # Number of different outputs (Dog, Cat, Lion...) as opposed to features (Color, Weight, Length)
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_examples < self.min_examples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split

        # n_features - the total number of features; self.n_features - the number of features we want to select; replace=False excludes the duplicates from being chosen. Only unique features are going to be in the 1Darray.
        # So, if the n_features is 30, and the self.n_features is 15, then we pick 15 random integers (non-repeating, as replace=False) out of the 30 integers. (0, 1, 2, ..., 30)
        # These integers are going to be the indices of the features or the indices of the columns of these features.
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:,best_feature], best_thresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        # What order are we going to return nodes in?
        # repeat recursion

        


    def _best_split(self, X, y, feat_idxs):
        """
        Get the index of a best feature to split on.
        Get the best threshold to split on in a given array of values for given feature.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            # Let's say the values of each example/row for this column are [1, 5, 7, 20, 43, 23]
            # Make those numbers to be thresholds
            # These are going to be used for comparing the information gain for every single one of those thresholds
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Calculate the information gain
                gain = self._information_gain(X_column, y, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    

    def _information_gain(self, X_column, y, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)


        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the weighted (average) entropy of children

        n_total_examples = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[n_left]), self._entropy(y[n_right])

        child_entropy = ((n_left / n_total_examples) * (e_left) + (n_right / n_total_examples) * (e_right))

        # Calculate the IG

        # The less the entropy of the child nodes, the more is the information gain.
        ig = parent_entropy - child_entropy

        return ig

    def _split(self, X_column, split_thresh):
        # the indices of the values that are less than the threshold (np.argwhere() returns a list of indices of values that are less than a given value in a given list)
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        return left_idxs, right_idxs


    def _entropy(self, y):
        # Count the number of occurences of each number starting from 0 up until the last int in list included
        # So, for list y = [1, 2, 3, 1, 2], it returns [0, 2, 2, 1]
        hist = np.bincount(y)

        # For list y = [1, 2, 3, 1, 2], len(y) = 5
        # So, ps = [0/5, 2/5, 2/5, 1/5] = [0, 0.4, 0.4, 0.2]
        # ps always add up to 1.    (Every p is p(x) = #x/n or the number of occurences of x divided by the total number of examples)
        ps = hist / len(y)

        # We use log because, for the range of values 0 < n < 1, the closer the given p is to 1, the closer is log2(1) to 0. The closer the entropy is to 0, the more clean the dataset is. (As at the extreme of 0, it means that p is 1,
        # and if p is 1, then there's only one p, or, in other words, one type of label)
        # Think if this is correct at all: (Indicating that the more expected the result is, the less information it brings.)
        # We use log2 because the output is binary. One binary decision is one bit.
        return -np.sum([-p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)

        # 1 - get only the most common element (if it was, for example, 2, then it means get the two most common elements); returns a tuple of an element and the number of occurences.
        # The first [0] - get the first tuple, as it is a list of tuples. (It just happens in this case that this list has a length of 1 as previously specified)
        # The second [0] - get the first element of the tuple, or the value of the most common label.
        most_common_label = counter.most_common(1)[0][0]

        return most_common_label