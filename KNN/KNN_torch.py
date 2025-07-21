import torch

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def predict(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        distances = torch.cdist(X_test, self.X_train)  # Euclidean distance
        knn_indices = distances.topk(self.k, largest=False).indices
        knn_labels = self.y_train[knn_indices]  # (num_test, k)
        predictions = torch.mode(knn_labels, dim=1).values
        return predictions.numpy()
