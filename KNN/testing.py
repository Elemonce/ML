import numpy as np

# lst = [[(np.int64(1), 3), (np.int64(2), 2)], [(np.int64(1), 3), (np.int64(2), 2)], [(np.int64(2), 3), (np.int64(1), 2)], [(np.int64(0), 5)], [(np.int64(1), 5)], [(np.int64(0), 5)], [(np.int64(0), 5)], [(np.int64(0), 5)], [(np.int64(1), 4), (np.int64(2), 1)], [(np.int64(2), 5)], [(np.int64(1), 4), (np.int64(2), 1)], [(np.int64(0), 5)], [(np.int64(2), 5)], [(np.int64(1), 4), (np.int64(2), 1)], [(np.int64(0), 5)], [(np.int64(1), 5)], [(np.int64(2), 5)], [(np.int64(0), 5)], [(np.int64(2), 5)], [(np.int64(1), 4), (np.int64(2), 1)], [(np.int64(1), 3), (np.int64(2), 2)], [(np.int64(1), 3), (np.int64(2), 2)], [(np.int64(2), 4), (np.int64(1), 1)], [(np.int64(1), 4), (np.int64(2), 1)], [(np.int64(2), 5)], [(np.int64(0), 5)], [(np.int64(2), 3), (np.int64(1), 2)], [(np.int64(1), 4), (np.int64(2), 1)], [(np.int64(2), 5)], [(np.int64(0), 5)]]

# numbers = [[1, 2], [3, 4]]

# print(lst[0])


ns = [1, 3, 4, 2, 7, 1, 0, 9, 11, 5]

print(np.argsort(ns))