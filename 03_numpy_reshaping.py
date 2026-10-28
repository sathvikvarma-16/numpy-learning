import numpy as np
"""
Numpy Reshaping 
Many beginners understand the values inside an array but get confused when a model gives an error such as:
Expected 2D array, got 1D array instead
The problem is often not the data itself, The problem is the shape.

For your ML Engineering, you should become very comfortable with:
1D array
   ↓
reshape
   ↓
2D array
   ↓
transpose
   ↓
add/remove dimensions
   ↓
prepare data for ML model


"""

# RESHAPING     
arr = np.array([1,2,3,4,5,6])
print(arr) # [1 2 3 4 5 6]
print(arr.shape) # (6,) -> 6 elements 1 dimension 

""" now we can reshape it into 2 rows and 3 columns """
arr2 = arr.reshape(2,3)
print(arr2) # [[1 2 3] [4 5 6]]
print(arr2.shape) # the data didn't change, only the structure changes i.e (2,3)

"""
the most important rule of reshape is the elements must remain the same
we can reshape it into (2,3),(3,2),(1,6),(6,1) because rows X columns = number of elements
if no. of elements = 6, then we cannot reshape it into (4,2) since 4*2!=6, we will get an error
the fundamental rule in reshaping is original number of elements = new number of elements
"""

# Understanding (n,) => 1d array
print(np.array([1,2,3,4,5]).shape) # (5,) 
""" this means not 5 rows, means one dimension containing 5 elements -> all these are of type 1d arrays """

# understanding (n,1) -> means a column => column vector
print(np.array([1,2,3,4]).reshape(4,1).shape)  # 4 rows, 1 column

# understanding (1,n) -> means a row => row vector
print(np.array([1,2,3,4,5]).reshape(1,5)) # 1 row, 5 columns


# reshaping a 2d array
print(np.array([[1,2,3],[4,5,6]]).reshape(3,2)) #(2,3)->(3,2)
print(np.array([[1,2,3],[4,5,6]]).reshape(1,6)) # #(2,3)->(1,6)

# using -1 in reshape, numpy automatically calculates and reshapes the array
# here numpy automatically calls the missing dimension, can only used for 1 one unknown dimension
print(np.array([1,2,3,4]).reshape(2,-1)) # given two rows it automatically found columns to be 2 since 4/2 = 2
print(np.array([1,2,3,4,5,6,7,8]).reshape(4,-1)) # 8/4 = 2 
print(np.array([1,2,3,4,5,6,7,8]).reshape(-1,4)) # 4 columns finds 2 rows to be 
print(np.array([[1,2,3],[4,5,6]]).reshape(-1)) # [1 2 3 4 5 6] -> gives 1d array (flattening method)

# reshape doesn't change the original array automatically, the original array structure (shape) of array remains same



# reshape() vs resize()

"""
reshape() -> changes the shape while preserving the number of elements
resize() -> can change the total number of elements
"""
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8]) 
arr.resize(3, 2) # resize removes extra elements
print(arr) # [[1 2] [3 4] [5 6]] -> shape (3,2)
arr.resize(4, 3) # resize adds extra elements
print(arr) # [[1 2 3] [4 5 6] [7 8 9] [10 11 12]] -> shape (4,3)


# ravel -> converts an array into an 1d array
print(np.array([[1,2,3],[4,5,6],[7,8,9]]).ravel()) # [1 2 3 4 5 6 7 8 9] -> shape (3,3)->(9,)
# flatten -> also converts into an 1d array -> but always returns a copy
print(np.array([[12,22,32],[42,52,62],[72,82,92]]).flatten()) # [12 22 32 42 52 62 72 82 92] 


# difference b/w ravel() and flatten()
"""
ravel almost similar to view, changes are affected in the original array but flatten always returns a copy!
Use ravel() when you want efficiency.
Use flatten() when you explicitly need an independent copy.
"""

# transpose of a matrix
# the same operation can be done with np.transpose(matrix)
# transpose of a matrix cannot works with 1d, becoz explictly it doesnt have any rows or columns
print(np.array([[1,2,3],[4,5,6]]).T) # shape (2,3)->(3,2)


"""
np.expand_dims()
expand_dims() adds a new dimension

axis=0 → add a row dimension
axis=1 → add a column dimension
"""

arr = np.array([1,2,3,4,5]) # shape : (5,)
new = np.expand_dims(arr,axis=0) # adding dimension to create row
print(new.shape) # shape : (1,5)
print(new)  # [[1 2 3 4 5]]  
new = np.expand_dims(arr,axis=1)  # since it was 1d array, adds a dimension to create a column 
print(new.shape) # (5,1) 5 rows, 1 column
print(new)  # output : [[1] [2] [3] [4] [5]]



# equivalent to add rows and columns
arr = np.array([1,2,3,4,5])
print(arr) # [1 2 3 4 5]
print(arr.shape) # (5,)
print(arr.reshape(1,-1)) # [[1 2 3 4 5]] -> given we want 1 row hence columns are automatically calculated to be 5
print(arr.reshape(-1,1)) # [[1] [2] [3] [4] [5]] -> given we want 1 column hence rows are automatically calculated to be 5
print(np.expand_dims(arr, axis=0)) # [[1 2 3 4 5]] -> adds a row
print(arr[np.newaxis,:]) # [[1 2 3 4 5]]
print(arr[None,:]) # [[1 2 3 4 5]] -> None acts as an alias for np.newaxis(), It inserts a new dimension of size 1 at the specified position (index 0 here).
print(arr[:,None]) # [[1] [2] [3] [4] [5]] -> adds a column
print(arr.reshape(-1,1))  # [[1]/ [2]/ [3]/ [4]/ [5]] -> given we want 1 columns here it automatically calculates rows to be 5
print(np.expand_dims(arr, axis=1))  # [[1]/ [2]/ [3]/ [4]/ [5]] -> axis=1 refers to the second dimension (columns), so it adds a new column dimension.
print(arr[:,np.newaxis]) # [[1]/ [2]/ [3]/ [4]/ [5]] -> here np.newaxis is used to add a new axis (dimension) to the array, effectively converting it from a 1D array to a 2D column vector.
print(arr[:,None]) # [[1]/ [2]/ [3]/ [4]/ [5]]




# Important squeeze() Rule -> squeeze() removes dimensions whose size is exactly 1.
arr = np.array([
    [1],
    [2],
    [3]
]) # shape (1,3)
result = np.squeeze(arr) #  (3,)
arr = np.array([
    [1, 2, 3] # shape (1,3)
])
result = np.squeeze(arr) # (3,)

""" 
squeeze() rule :
(1, 3) → (3,)
(3, 1) → (3,)
(1, 3, 1) → (3,)
"""



# 3D ARRAYS - understand 3d reshaping conceptually
arr = np.arange(24) 
# arr.shape = (24,)
# reshaping
print(arr.reshape(2,4,3)) # 3rd matrix of 2 blocks, 4 rows, 3 columns
print(arr.reshape(3,4,2)) # 3d matrix of 3 blocks, 4 rows, 2 columns
print(arr.reshape(2,3,4)) # 3d matrix with 2 blocks, 3 rows, 4 columns



# Image Example
"""
a grayscale image with 28X28 pixels can be represented as : (28,28)
there are : 782 pixels
you can flatten it : 
"""
image = np.zeros((28,28))
flatten = image.reshape(-1)
print(flatten) # shape is : (28,) 
image = flatten.resize(28,28) # can again restore it


#  RGB image example
"""
height = 224
width = 224
channels = 3
shape : (224,224,3)
this represents : (height X width X channel)
for a batch of 32 innings : (32,224,224,3) 
meaning :
32 images
224 height
224 width
3 color channels

understanding dimensions is essential for future Deep Learning.
"""



# Reshaping database data - common ml operation
data = np.array([[1,2,3],[4,5,6],[7,8,9]]) # shape : (3,3) meaning => 3 features, 3 labels
features = data[:,0] # extract the first feature => shape ()
print(features.shape) # 3
features = features.reshape(-1,1) # need to reshape with 1 column, numpy calculates the number of rows
print(features) # [1 4 7] -> shape : (3,1) 3 rows, 1 column



# important concept -
"""
(-1, 1) → many rows, one column
(1, -1) → one row, many columns
"""



# important difference .T and reshape(1,-1)
# don't use .T to convert 1d array to row or a column
print(np.array([1,2,3,4]).reshape(-1,1).shape) # since shape of array is : (4,)after reshaping its shape is : (4,1)
# But. in case of .T it would not be like this the shape doesn't changes
print(np.array([1,2,3,4]).T.shape) # (4,)




# order in reshape
# In reshape(), order tells NumPy how to read the original elements and place them into the new shape.

arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])   # shape : (2, 3)
# order='C' : Row-major order ->This is the default
arr.reshape(3, 2, order='C')  # NumPy reads the elements row by row: 1 2 3 4 5 6

""" 
filling of new shape : 
[[1, 2],
 [3, 4],
 [5, 6]]

C = C-style = row by row
"""


# order='F' : Column-major order
arr.reshape(3, 2, order='F') # NumPy reads the original array column by column: 1 4 2 5 3 6

"""
Then fills the new shape:
[[1, 4],
 [2, 5],
 [3, 6]]

F = Fortran-style = column by column
"""


# usually reshape returns A View
arr = np.array([1, 2, 3, 4])
matrix = arr.reshape(2, 2)
matrix[0, 0] = 999
print(arr) # [999 2 3 4] -> This happens because the reshaped array may share memory with the original

# If you need an independent copy:
matrix = arr.reshape(2, 2).copy()  # -> independent data => no changes in original array

