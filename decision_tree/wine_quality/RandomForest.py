from Tree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_examples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_examples_split = min_examples_split
        self.n_features = n_features
        self.trees = None

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_examples_split=self.min_examples_split)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def _bootstrap_samples(self, X, y):
        n_examples = X.shape[0]

        idxs = np.random.choice(n_examples, n_examples, replace=True)

        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def predict(self, X): 
        # A list of lists for each tree where each element is a prediction for each example
        # [[1, 0, 1, 1], [0, 1, 1, 1,], [0, 1, 1, 0]]
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        tree_preds = np.swapaxes(predictions, 0, 1)
        
        return np.array([self._most_common_label(pred) for pred in tree_preds])
    


from sklearn import datasets
from sklearn.model_selection import train_test_split
    
data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return accuracy
    



    

    

    


        

    