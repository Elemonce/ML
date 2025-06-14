# import numpy as np
# test = np.array([0, 1, 2, 3])

# lst = np.array([43, 342, 234, 4,34, 2,4,2343,43,23,23, 2])

# print(lst[test])


# def fibonacci(n, iteration):
#     if n <= 1:
#         print(f"iteration number: {iteration}")
#         return n
    
#     return fibonacci(n - 1, iteration+1) + fibonacci(n - 2, iteration+1)
#     # return fibonacci(n - 2) + fibonacci(n - 1)

# print(fibonacci(4, 0))


def factorial(n):
    print(n)
    if n <= 1:
        return n
    
    return factorial(n - 1) * n

print(factorial(5))