import numpy as np

class LinearRegression:
    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters


    def fit(self, X, y):
        m, n = X.shape
        self.w_ = np.zeros(n)
        self.b = 0


        # column vector is of shape (n, 1)
        # row vector is of shape (1, n)

        # X: (m, n), w_: (n, 1)
        # return type: (m, 1) matrix
        f_wb = np.dot(X, self.w_) + self.b

        # X.T: (n, m); err: (m, 1)
        # return type: (n, 1) matrix
        err = f_wb - y
        dj_dw = np.dot(X.T, err) / m

