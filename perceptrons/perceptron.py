import numpy as np
class Perceptron:
    """Perceptron classifier.
    
    Parameters
    --------------
    eta: float
        Learning rate (between 0.0 and 1.0)

    n_iter: int
        Passes over the training dataset.

    random_state: int
    Random number generator seed for random weight initialization.

    Attributes
    --------------
    w_ : 1d-array
        Weights after fitting.

    b_ : Scalar
        Bias unit after fitting. 

    errors_ : list
    Number of missclasifications (updates) after each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        # Doesn't matter how big the eta is here as long as it's not too small and isn't negative
        # because we classify the prediction as 1 when the value is >= 0, so it doesn't matter how big or small
        # (even negative) the number is.
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        Parameters 
        -------------
        
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of 
            examples and n_features is the number of features.

        y : array-like, shape = [n_examples]
            Target values.

        Returns 
        -------
        self : object
        """

        # Use the seed to generate pseudo-random numbers
        # (Pseudo because a given seed always returns the same results up to roundoff error,
        # except when the values were incorrect.)
        rgen = np.random.RandomState(self.random_state)

        # loc - mean or the "centre" of the distribution.
        # scale - standard deviation (spread or width) of the distribution.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # self.w_ = np.zeros(X.shape[1])
        self.b_ = np.float64(0.) 
        # self.b_ = 0.0
        self.errors_ = []

        update = None
        for i in range(self.n_iter):
            # print("Iteration", i)
            # print("w:", self.w_, ", b:", self.b_)
            # print("Update: ", update)
            errors = 0
            # print(len(X))
            # print(len(y))
            # print(list(zip(X, y))[0])
            for x_i, target in zip(X, y):
                # print("X shape", X.shape)
                # print("X[0]", X[0])
                # print(xi, target)


                # If self.predict returns 1 and target is 0, then update is -self.eta, so that the values of self.w_ and self.b_ 
                # become a bit lower and gradually reach the threshold needed to predict a 0 rather than 1.
                # Same principle for when self.predict returns 0 and target is 1.
                # If predict and target are equal, update is 0 and no changes happen.
                update = self.eta * (target - self.predict(x_i))


                init = self.w_.copy()
                # x_i is a row of X. update the vector of w by adding the elements of vector x_i times update. 
                # (element by element operation)
                self.w_ += update * x_i
                self.b_ += update

                if np.all(init == self.w_) == False:
                    print("update, xi, b:", update, x_i, self.b_)
                # If update is 0, converts False to 0 and adds no error. If update is 1 
                # (because it can only be 1 here, this being a classification algorithm with 1s and 0s), then 
                # adds an error.
                errors += int(update != 0.0)
            # For every iteration, add how many errors occurred during that iteration
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        # print("net input", np.dot(X, self.w_) + self.b_)
        # print(X.shape, self.w_.shape)

        # returns a negative value in case w is randomly initialized to a negative number. 
        # however, it works even if all w's are initialized to zero. why?

        # It works because net input of 0 is going to make a prediction of 1 for predict method. 
        # Then, the fit method adjusts the values using the update variable. 
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Return class label after unit step.""" # Unit step is a function that returns either 1 or 0 based on whether 
        # a given examples passes a given threshold. 

        # Second argument, 1, is when condition evaluates to True.
        # Third is when it evaluates to False.


        # always returns an ndarray of size 1 because the return value of self.net_input(X) is a scalar
        # print("predict type: ", type(np.where(self.net_input(X) >= 0.0, 1, 0)))
        # print("predict shape: ", (np.where(self.net_input(X) >= 0.0, 1, 0)).shape)
        # print("predict value: ", np.where(self.net_input(X) >= 0.0, 1, 0))

        print("shape of X given to the method: ", X.shape)
        # print("value: ", np.where(self.net_input(X) >= 0.0, 1, 0), "type: ", type(np.where(self.net_input(X) >= 0.0, 1, 0)), "shape: ", (np.where(self.net_input(X) >= 0.0, 1, 0)).shape)
        print("shape of the return value of the method: ", np.where(self.net_input(X) >= 0.0, 1, 0).shape)
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    





