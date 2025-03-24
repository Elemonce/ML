Number of examples
```python
print(X.shape[0])
```

Number of features
```python
print(X.shape[1])
```

# Range
max_val - min_val: The range of elements. It defines how far apart the smallest and the largest values are.
To calculate it, apply the same principle as for X - min_val, except that there is only one example of max_val
and min_val, so min_val is just substracted from max_val element-wise.