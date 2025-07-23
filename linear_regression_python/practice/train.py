import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)




regression = LinearRegression(alpha=0.1, n_iters=1000)

regression.fit(X_train, y_train)

predictions = regression.predict(X_test)
print(y_test.shape)
print(regression.cost(y_test, predictions))

# print(regression.cost(X_train, y_train))

# regression.show_cost_history()




# y_pred_line = predictions
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# train = plt.scatter(X_train[:,0], y_train, color=cmap(0.9), marker="x", s=30)
# test = plt.scatter(X_test[:,0], y_test, color=cmap(0.5), marker="x", s=30)
# # plt.plot(X[:20, 0], y_pred_line, color="black", linewidth=2, label="Prediction")
# plt.plot(X_test, y_pred_line, color="black", linewidth=2, label="Prediction")
# # plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

regression.show_cost_history()