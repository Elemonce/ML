import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters


    def fit(self, X, y):
        m, n = X.shape
        self.w_ = np.zeros(n)
        self.b_ = 0


        # column vector is of shape (n, 1)
        # row vector is of shape (1, n)

        self.cost_hist_ = []

        for _ in range(self.n_iters):
            # X: (m, n), w_: (n, 1)
            # return type: (m, 1) matrix
            f_wb = np.dot(X, self.w_) + self.b_

            # X.T: (n, m); err: (m, 1)
            # return type: (n, 1) matrix
            err = f_wb - y
            dj_dw = np.dot(X.T, err) / m
            dj_db = np.sum(err) / m

            self.w_ -= self.alpha * dj_dw
            self.b_ -= self.alpha * dj_db

            cost = self._compute_cost_for_graph(X, y)
            self.cost_hist_.append(cost)




    def predict(self, X):
        predictions = np.dot(X, self.w_) + self.b_
        print("prediction shape", predictions.shape)
        return predictions
    

    def _compute_cost_for_graph(self, X, y):
        cost = 0
        m = X.shape[0]
        for i in range(m):
            f_wb_i = np.dot(X[i], self.w_) + self.b_
            loss = (f_wb_i - y[i]) ** 2

            cost += loss

        return cost / (2 * m)
    
    def cost(self, y_test, predictions):
        return np.mean((y_test - predictions) ** 2) / 2
    

    def show_cost_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
            
        ax1.plot(self.cost_hist_)
        ax2.plot(100 + np.arange(len(self.cost_hist_[:100])), self.cost_hist_[:100])
        ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
        ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
        ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
        plt.show()
