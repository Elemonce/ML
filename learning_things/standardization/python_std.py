def mean_std(lst):
    mean = sum(lst) / len(lst)

    # for x in lst, substract mean from it and square it 
    # sum the received values and divide them by the length of the list.
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)

    # square root of variance
    std = variance ** 0.5
    return mean, std

def standardize_pure(X):
    means = [mean_std(col)[0] for col in zip(*X)]
    # for e in zip(*X):
    #     print(e)
    stds = [mean_std(col)[1] for col in zip(*X)]
    
    standardized_X = []
    for row in X:
        standardized_row = [(val - mean) / std if std != 0 else 0
                            for val, mean, std in zip(row, means, stds)]
        standardized_X.append(standardized_row)

    return standardized_X

# Example usage
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

X_standardized = standardize_pure(X)
print(X_standardized)
