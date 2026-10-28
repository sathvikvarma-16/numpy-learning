import numpy as np


# Problem 1: Create a NumPy array
a = np.array([10, 20, 30, 40, 50])
print(a) # [10,20,30,40,50]


# Problem 2: Check datatype and shape
a = np.array([10, 20, 30])
print(a.dtype) # int 64 
print(a.shape) # (3,)


# Problem 3: Create zeros, ones and a range
print(np.zeros(4)) # [0. 0. 0. 0.]
print(np.ones(4)) # [1. 1. 1. 1.]
print(np.arange(1, 6)) # [1 2 3 4 5]


# Problem 4: Create a 2D array
a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(a)

# [[1 2 3]
#  [4 5 6]]


# Problem 5: Access elements
a = np.array([10, 20, 30, 40])

print(a[0]) # 10
print(a[-1]) # 40


# Problem 6: Access elements in a 2D array
a = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print(a[0, 1]) # 20
print(a[1, 2]) # 60



# Problem 7: Slice an array
a = np.array([10, 20, 30, 40, 50])

print(a[1:4]) # [20 30 40]
print(a[:3]) # [10 20 30]
print(a[::2]) # [10 20 30]


# Problem 8: Slice a 2D array
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(a[:2, 1:])

# [[2 3]
#  [5 6]]


# Problem 9: Change an element
a = np.array([10, 20, 30])
a[1] = 99

print(a) # [10 99 30]


# Problem 10: Reshape an array
a = np.arange(1, 7)
b = a.reshape(2, 3)

print(b)

# [[1 2 3]
#  [4 5 6]]


# Problem 11: Flatten a 2D array
a = np.array([
    [1, 2],
    [3, 4]
])

print(a.flatten()) # [1,2,3,4]


# Problem 12: Check dimensions and size
a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(a.ndim) # 2
print(a.size) # 6


# Problem 13: Add two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b) # [5,7,9]


# Problem 14: Perform element-wise operations
a = np.array([10, 20, 30])

print(a * 2) # [20 40 60]
print(a + 5) # [15 25 35]
print(a ** 2) # [100 400 900]


# Problem 15: Broadcasting
a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(a + 10)

# [[11 12 13]
#  [14 15 16]]


# Problem 16: Column broadcasting
a = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

b = np.array([10, 20])

print(a + b)

# [[11 22]
#  [13 24]
#  [15 26]]


# Problem 17: Boolean filtering
a = np.array([10, 25, 5, 40, 15])

print(a[a > 20]) # [25,40]


# Problem 18: Multiple conditions
a = np.array([10, 20, 30, 40, 50])
print(a[(a > 15) & (a < 45)]) # [20 30 40]


# Problem 19: Replace values using a condition
a = np.array([10, 20, 30, 40])
a[a > 25] = 0
print(a) # [10 20  0  0]


# Problem 20: Minimum, maximum and mean
a = np.array([10, 20, 30, 40, 50])
print(a.min()) # 10
print(a.max()) # 50
print(a.mean()) # 30.0



# Problem 21: Sum and standard deviation
a = np.array([10, 20, 30, 40])

print(a.sum()) # 100
print(a.std()) #  11.180339887498949


# Problem 22: Statistics by axis
a = np.array([
    [10, 20],
    [30, 40]
])
 
print(a.sum(axis=0)) # [40 60]
print(a.sum(axis=1)) # [30 70]


# Problem 23: Sort an array
a = np.array([40, 10, 30, 20])

print(np.sort(a)) # [10,20,30,40]


# Problem 24: Find indices of sorted elements
a = np.array([40, 10, 30, 20])

print(np.argsort(a)) # [1,3,2,0]



# Problem 25: Find values and positions
a = np.array([10, 50, 20, 80, 30])
print(np.argmax(a)) # 3
print(np.argmin(a)) # 0



# Problem 26: Search for a value
a = np.array([10, 20, 30, 40, 50])
print(np.where(a == 30))

# Output:
# (array([2]),)


# Problem 27: Combine arrays
a = np.array([1, 2])
b = np.array([3, 4])
print(np.concatenate((a, b))) # [1,2,3,4]


# Problem 28: Stack arrays vertically and horizontally
a = np.array([1, 2])
b = np.array([3, 4])
print(np.vstack((a, b)))
print(np.hstack((a, b)))

# [[1 2]
#  [3 4]]
# [1 2 3 4]


# Problem 29: Split an array
a = np.array([1, 2, 3, 4, 5, 6])
print(np.split(a, 3)) # [array([1, 2]), array([3, 4]), array([5, 6])]


# Problem 30: Handle NaN values
a = np.array([10, np.nan, 30, np.nan])
print(np.isnan(a)) # [False  True False  True]
print(np.nanmean(a)) # # 20.0



# Problem 31: Remove NaN values
a = np.array([10, np.nan, 30, np.nan])
clean = a[~np.isnan(a)]
print(clean) # [10. 30.]


# Problem 32: Handle infinity
a = np.array([10, np.inf, 30, -np.inf])
print(np.isfinite(a)) # # [ True False  True False]


# Problem 33: Generate random numbers
np.random.seed(1)
print(np.random.randint(1, 10, 5)) # [6 9 6 1 1]


# Problem 34: Random choice
np.random.seed(1)
print(np.random.choice([10, 20, 30], 5)) # [20 10 10 20 20]


# Problem 35: Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b)) # 32


# Problem 36: Matrix multiplication
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A @ B)

# [[19 22]
#  [43 50]]


# Problem 37: Element-wise multiplication vs matrix multiplication
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A * B)
print(A @ B)

# [[ 5 12]
#  [21 32]]
# [[19 22]
#  [43 50]]


# Problem 38: Transpose
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.T)

# [[1 4]
#  [2 5]
#  [3 6]]


# Problem 39: Identity matrix
print(np.eye(3))
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]


# Problem 40: Matrix inverse
A = np.array([
    [1, 2],
    [3, 4]
])
print(np.linalg.inv(A))

# [[-2.   1. ]
#  [ 1.5 -0.5]]


# Problem 41: Determinant
A = np.array([
    [1, 2],
    [3, 4]
])

print(np.linalg.det(A)) # -2.0


# Problem 42: Solve a system of equations
A = np.array([
    [2, 1],
    [1, 1]
])

b = np.array([5, 3])
x = np.linalg.solve(A, b)
print(x) # [2. 1.]


# Problem 43: Eigenvalues and eigenvectors
A = np.array([
    [2, 0],
    [0, 3]
])
values, vectors = np.linalg.eig(A)
print(values) # [2. 3.]


# Problem 44: Matrix rank
A = np.array([
    [1, 2],
    [2, 4]
])
print(np.linalg.matrix_rank(A)) # 1


# Problem 45: Vector norm
a = np.array([3, 4])
print(np.linalg.norm(a)) # 5.0


# Problem 46: ML dataset -> X and y
data = np.array([
    [20, 20000, 0],
    [25, 30000, 1],
    [30, 40000, 1]
])

X = data[:, :2]
y = data[:, 2]

print(X)
print(y)

# Output:
# [[   20 20000]
#  [   25 30000]
#  [   30 40000]]
# [0 1 1]


# Problem 47: Check samples and features
print(X.shape)
print(y.shape)

print("Samples:", X.shape[0])
print("Features:", X.shape[1])

# Output:
# (3, 2)
# (3,)
# Samples: 3
# Features: 2


# Problem 48: Convert 1D data into 2D feature data
X = np.array([20, 25, 30])

X = X.reshape(-1, 1)

print(X)
print(X.shape)

# Output:
# [[20]
#  [25]
#  [30]]
# (3, 1)


# Problem 49: Normalize data
salary = np.array([20000, 30000, 40000])

normalized = (
    (salary - salary.min())
    / (salary.max() - salary.min())
)

print(normalized)

# Output:
# [0.  0.5 1. ]


# Problem 50: Vectorized ML calculation
X = np.array([
    [1, 10],
    [2, 20],
    [3, 30]
])

weights = np.array([2, 3])

result = X @ weights

print(result)

# Output:
# [32 64 96]


# Problem 51: Complete small ML preprocessing example
data = np.array([
    [20, 20000, 0],
    [25, 30000, 1],
    [30, 40000, 1],
    [35, 50000, 0]
])

X = data[:, :2]
y = data[:, 2]

# Check shape
print(X.shape)
print(y.shape)

# Normalize features
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_normalized = (X - X_min) / (X_max - X_min)

print(X_normalized)
print(y)

# Output:
# (4, 2)
# (4,)
# [[0.         0.        ]
#  [0.33333333 0.33333333]
#  [0.66666667 0.66666667]
#  [1.         1.        ]]
# [0 1 1 0]


# Final challenge:
# Given this dataset, separate X and y,
# find the average of each feature,
# and normalize X.

data = np.array([
    [10, 100, 1],
    [20, 200, 0],
    [30, 300, 1],
    [40, 400, 0]
])

X = data[:, :2]
y = data[:, 2]

print(X)
print(y)
print(X.mean(axis=0))

X_normalized = (X - X.min(axis=0)) / (
    X.max(axis=0) - X.min(axis=0)
)

print(X_normalized)

# Output:
# [[ 10 100]
#  [ 20 200]
#  [ 30 300]
#  [ 40 400]]
# [1 0 1 0]
# [ 25. 250.]
# [[0.         0.        ]
#  [0.33333333 0.33333333]
#  [0.66666667 0.66666667]
#  [1.         1.        ]]
```
