"""
Numpy - numerical python

It is a Python library used for:
Numerical calculations
Working with arrays
Matrix operations
Data manipulation
Statistics
Linear algebra
Machine Learning

You will use NumPy as a foundation for libraries such as:
Pandas
Scikit-learn
TensorFlow
PyTorch

"""
# a simple example of using numpy
import numpy as np
numbers = np.array([10, 20, 30, 40])
print(numbers) # [10 20 30 40] here np.array() creates an numpy array


"""
2. Why Do We Need NumPy?
Python already has lists.
numbers = [10, 20, 30, 40]

So why do we need NumPy?
Because NumPy is specifically designed for numerical computing.
For example, suppose we want to add 10 to every number.
"""
# without numpy
numbers1 = [10, 20, 30, 40]
result = []
for number in numbers1:
    result.append(number + 10)
print(result)

# with numpy
import numpy as np
numbers2 = np.array([1,2,34,5])
print(numbers2+10)

# Vectorized operation, Instead of manually writing a loop, NumPy performs the operation on the entire array.
# This is one of the most important ideas in NumPy.
# NumPy arrays are designed primarily for numerical data.

print(type(numbers1))  # list
print(type(numbers2)) # <class 'numpy.ndarray'>   => this is the main data structure of python

# ndarray -> n dimensional array 
# numpy can work with many dimensions (1d,2d,3d)

# 2d array printing
matrix = np.array([[1,2,3],[4,5,6]])
print(matrix)


# np.zeros(n) -> create an array of size n with element as 0
zeros = np.zeros(10) # one dimensional
print(zeros) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# 2d zeroes
zeros1 = np.zeros([2,3]) # 2 rows, 3 columns
print(zeros1)
zeros2 = np.zeros([3,2]) # 3 rows, 2 columns
print(zeros2)

# .shape() -> .shape tells you the structure or dimensions of an array.
image = np.zeros([28,28])
print(image.shape) # return (28,28) this means (rows, columns)


# np.ones(n)
print(np.ones(5)) # [1. 1. 1. 1. 1.]
print(np.ones((2,4)))  # 2 rows 4 columns
print(np.empty(5)) # It means NumPy allocates memory without setting initial values, The output may contain arbitrary values.


# np.full(length,number) -> Creates an array filled with a specific value.
print(np.full(5,7))  # [7 7 7 7 7] 
print(np.full(2,100))  # [100 100]
print(np.full((2,3),10))  # matrix of 2 rows, 3 columns with numbers as 10


"""
np.arange()
This is extremely important.
It works similarly to Python's range().
"""

print(np.arange(10)) # [0 1 2 3 4 5 6 7 8 9]
print(np.arange(2,10)) # (start,stop) [2 3 4 5 6 7 8 9]
print(np.arange(0,10,2)) # (start, stop, step) [0 2 4 6 8]
print(np.arange(10,0,-2)) # [10 8 6 4 2]

"""
np.linspace(start,stop,number of values)
It creates exactly (number of values) values between start and stop
np.linspace() creates a specific number of equally spaced values.
"""

print(np.linspace(0,10,3)) # [ 0.  5. 10.]
print(np.linspace(9,100,5)) # [  9.    31.75  54.5   77.25 100.  ]


"""
arange() → step-based, stop usually excluded
linspace() → number-of-values-based, stop included by default
"""

# np.eye(n) -> creates an identity matrix of order n
print(np.eye(5)) # identity matrix of order 5 
print(np.eye(2)) # identity matrix of order 2 


# Array properties

# 1. ndim - returns the number of dimensions
arr = np.array([[1,2],[3,4]])
print(arr.ndim) # returns the number of dimensions -> 2
print(np.array([1,2,3]).ndim) # 1 because it was 1 dimensional array

# 2. shape -> Returns the size of every dimension (important for ml)
print(np.array([[1,2],[3,4],[5,6]]).shape)  #(3,2) 3 rows, 2 columns

# 3. size - returns the total number of elements 
# if 3 rows and 2 columns => 3X2 = 6
print(np.array([[1,2,3],[4,5,6]]).size) # 6

# 4. dtype - returns the datatype
print(np.array([1,2,34,55]).dtype) # int32
print(np.array([1.23,4.55,6.33,6]).dtype) # float64
ar1 = np.array([1,2,3],dtype=float)  # converts to float
ar2 = np.array([1.2,3.4],dtype=int) # the decimal part is removed => converts to int
print(ar1)
print(ar1.dtype) # float64
print(ar2)
print(ar2.dtype) # int32

# 5. itemsize - returns the number of bytes used by the each element -> This depends on the data type.
print(np.array([1,23,45]).itemsize) 

# important concept - shape
print(np.array([1,2,3,4]).shape) # (4,) i.e 4 elements in 1 dimension
print(np.array([[1,2,3,4]]).shape) # (1,4) i.e 1 row 4 columns


"""
Data Type Conversion
You can convert an array using:
.astype()
"""
a = np.array([1,2,3,4])
b = a.astype(float)
print(b)
print(b.dtype) # float

a1 = np.array([.23,3.445])
b1 = a1.astype(int)
print(b1) # int
print(b1.dtype)


# Numpy arrays and mathematical operations 
print(np.array([1,2,3,4])+np.array([1,2,3,4])) # matrix addition
print(np.array([[1,2],[3,4],[5,6]])+np.ones((3,2))) # matrix multiplication   # (1X3)x(3X2) => (1X2)

# Create 10 equally spaced values between 0 and 1 using np.linspace().
print(np.linspace(0,1,10)) # output : [0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556 0.66666667 0.77777778 0.88888889 1.  ]      

