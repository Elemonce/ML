import numpy as np

# def min_max_scaling(X, feature_range=(0, 1)):
#     min_val = X.min(axis=0)  # Minimum per feature
#     max_val = X.max(axis=0)  # Maximum per feature

#     X_scaled = (X - min_val) / (max_val - min_val)

#     # Rescale to the desired range if needed
#     min_range, max_range = feature_range
#     X_scaled = X_scaled * (max_range - min_range) + min_range

#     return X_scaled

def min_max_scaling(X):
    """
    min_val: An array of shape (X.shape[1]) where each element is the minimum value found for each feature column across m examples of X.

    max_val - Same principle as min_val but for maximum values.

    X - min_val: Subtract the values of min_val from each m of X using broadcasting.
    In other words, for each row m in X, substract min_val from every m element-wise.


    max_val - min_val: The range of elements. It defines how far apart the smallest and the largest values are.
    To calculate it, every feature of min_val is subtracted from every feature of max_val (Element-wise operation).

    (X - min_val) / (max_val - min_val): 
        For every m of X, subtract min_val from m using broadcasting.
        Divide each n of X[m] by (max_val - min_val)[n] using broadcasting.
    """
    min_val = X.min(axis=0)  # Minimum per feature
    max_val = X.max(axis=0)  # Maximum per feature
    
    X_scaled = (X - min_val) / (max_val - min_val)

    # print(f"Min value: {min_val}")
    # print(f"Max value: {max_val}")

    print(f"X - min_val: {X - min_val}")
    print(f"max_val - min_val: {max_val - min_val}")

    return X_scaled

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [19, 0, 0], [10, 11, 12]])

# Number of examples
# print(X.shape[0])

# # Number of features
# print(X.shape[1])
# X_scaled = min_max_scaling(X)
# print(X_scaled)

data = {
    'age': [18, 25, 40, 60],
    'income': [20000, 50000, 80000, 100000]
}
import pandas as pd
df = pd.DataFrame(data)

print(min_max_scaling(df))
