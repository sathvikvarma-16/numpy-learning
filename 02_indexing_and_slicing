import numpy as np
"""
Indexing and Slicing - This must be strong.

Indexing and slicing will be used constantly in:
NumPy
Pandas
Machine Learning datasets
Image processing
Neural networks
Data preprocessing

"""

# accessing, slicing, -ve indexing, +ve indexing
arr = np.array([1,2,3,4,5])
print(arr[0]) # 1
print(arr[1]) # 2
print(arr[-1]) # 5
print(arr[-2]) # 4

# 2d arrays
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print(matrix[1,1]) # 50
print(matrix[0,2]) # 30
print(matrix[-1,-1]) # -ve indexing == 90
print(matrix[-1,0])  # 70
print(matrix[0,:]) # select entire row (: means select everything in that dimension) [10 20 30]
print(matrix[:,0]) # all rows column 0 # [10 40 70]


# This is extremely important for Machine Learning. ex :-
x = np.array([
    [20, 50000],
    [25, 60000],
    [30, 80000]
])
ages = x[:,0]
salaries = x[:,1]
print(ages) # [20 25 30]
print(salaries) # [50000 60000 80000]

matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

print(matrix[0:2, 1:3]) # row 0,1 & cols 1,2 => [[20 30],[60 70]]

print(matrix[1:3, 1:3]) # [[ 60  70] [100 110]]

print(matrix[::-1,:]) # reversing rows [[ 90 100 110 120] [ 50  60  70  80] [ 10  20  30  40]]

print(matrix[:,::-1]) # reversing columns  [[ 40  30  20  10] [ 80  70  60  50] [120 110 100  90]]

print(matrix[::-1,::-1]) # reversing entire rows and columns  [[120 110 100  90] [ 80  70  60  50] [ 40  30  20  10]]

matrix[0] = [1,2,3,4] # modifying the rows simply

matrix[:,0]=1 # modifying columns -> for every row the 1st column would be 1

print(matrix)  # [[  1   2   3   4] [  1  60  70  80] [  1 100 110 120]]


# modifying a slice
nums = np.array([1,2,3,4])
nums[1:4]=0
print(nums) # [1 0 0 0]
nums[1:4]=[100,200,300] # nums -> [1 100 200 300]

# broadcasting during assignment
nums[1:4]=99 # nums -> [1 99 99 99 ] NumPy assigns 99 to every selected element.



# 3D arrays - Think of a 3D NumPy array as multiple 2D matrices stacked together.

arr = np.array([
    [
        [1, 2, 3],          # layer 0 | then after rows and columns
        [4, 5, 6]
    ],

    [
        [7, 8, 9],         # layer 1 | then after rows and columns
        [10, 11, 12]
    ]
])

# arr.shape -> (2,2,3)  => 2 layers, 2 rows, 3 columns
# the general pattern is : (number of layers, rows, columns) || (depth,height,width)
# ex : if arr[0,1,2] => layer 0, row 1, column 2 => element i.e 6


"""
real word example : 
grayscale images : images with height and width
image = (height,width)
image.shape  => # (28,28) one grayscale image is a 2d array
if we have 100 grayscale images => (100,28,28) means 100 images X 28 rows X 28 columns
"""

# ex : there is an 3d array
images = np.zeros((100,28,28))
print(images[0]) # you can access the first image
print(images[0,10,15]) # the pixel of the first image at 10th row & 15th column 


"""
RGB images are also 3D arrays

A color image usually has:
(height, width, channels)

For example : (224, 224, 3)

Means:
224 rows X 224 columns X 3 color channels

The three channels are:
Red
Green
Blue

So:
image.shape could be : (224, 224, 3)

Conceptually:
Red channel    → 224 × 224
Green channel  → 224 × 224
Blue channel   → 224 × 224


4D arrays in Machine Learning

Suppose you have:
32 color images
Each image is : 224 × 224 × 3

Together of 32 images looks like :
(32, 224, 224, 3)

This is a 4D array:
(number of images, height, width, channels)

This is extremely common in deep learning.


1D → (values)
2D → (rows, columns)
3D → (layers, rows, columns)
       OR
       (height, width, channels)
4D → (images, height, width, channels)

"""


"""
# INDEXING vs ACCESSING
arr[0]    → scalar value
arr[0:1]  → NumPy array
This can affect shapes in Machine Learning.
"""



# shape difference - 
arr = np.array([1,2,3,4])
print(arr.shape) # (4,)
print(arr[0:2].shape) # (2,) -> gives number of elements
print(arr[0].shape) #  A single number doesn't have shape => ()

matrix = np.array([[1,2,3,4],[5,6,7,8]])
print(matrix[0].shape) # it becomes a 1d array => (4,)
print(matrix[0:1].shape) # (1,4) => 1 row, 4 columns -> only changing rows not columns hence all columns will be print


# basic boolean indexing preview - although detailed filtering comes later, understand this connection
# this is called boolean indexing or boolean masking
arr = np.array([1,2,3,4,5,23,12,55,6,100])
low = arr[arr>25] # [arr>25] => [False,False,False,False,False,False,False,True,False,True]
print(low) # [55 100]



# selecting specific rows with index arrays
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix[[0,2]]) # rows 0, 2
print(matrix[[0,1]]) # rows 0, 1

# selecting specific columns
print(matrix[:,[0,2]]) # all rows with 0 and 2nd columns only
print(matrix[0:1,[1,2]]) # 0 row with 1st and 2nd columns only



# ML Example : Dataset Indexing  -> these are exactly the type of operations performed on databases

# [age,salary,experience]
dataset = np.array([[27,50000,3],[22,100000,1],[23,23500,2],[30,70000,4]])
print(dataset.shape) # (4,3) 4 rows, 3 columns 
ages = dataset[:,0] # select all ages
salaries = dataset[:,1] # select all salaries
experience = dataset[:,2] # select experience 
# select first two samples
print(dataset[0:2]) # == print(dataset[:2,:])
# select salary and experience
print(dataset[:,[1,2]]) # == print(dataset[:,1:3])


# ML example - features and labels
""" 
In a dataset, features and labels are two main parts in machine learning
features : input information used by the model to make a prediction. simply what the model see
eg : hours studied, attendance, previous marks

label : answer / target that model tries to predict, simply what the model predicts
ex : result ["Fail","Pass","Fail","Fail","Pass","Pass"]

Features(x) -> machine learning model -> label(y)

Ex : 
House size, bedrooms, location  → Features # input
House price                     → Label # prediction

Email words, sender, subject    → Features
Spam / Not Spam                 → Label

Image pixels                    → Features
Cat / Dog                       → Label


"""
data = np.array([
    [22, 50000, 0],
    [25, 60000, 1],
    [30, 80000, 1],
    [35, 90000, 1]
])
# first two columns features and last column label, we cam separate them
features = data[:,:2]
labels = data[:,2]
print(features)
print(labels)



# important concept : view vs copy
data = np.array([1,2,3,4])
part = data[1:]
part[0] = 100 # part = [100,3,4]
print(data) # [1 100 3 4] because slicing often returns a view of the original array rather than a completely independent copy.
# to create an independent copy 
part = data[1:].copy()
part[0]=200 # will not modify original array
print(part) # [200 3 4]
print(data) # [1 100 3 4]


# difference - arr[0] vs arr[[0]]
data = np.array([1,2,3,4])
print(data[0]) # returns scalar i.e 1
print(data[[0]]) # returns array [1]

data = np.array([[1,2,3,4],[5,6,7,8]])
print(data[0]) # prints as 1d array -> [1,2,3,4] 
print(data[0].shape) # (4,) # 1d array
print(data[[0]]) # prints entire row as 2d array -> [[1,2,3,4]]
print(data[[0]].shape) # (1,4)    # 2d array -> 1 row, 4 columns


