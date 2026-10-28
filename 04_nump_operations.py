import numpy as np

# Basic Arithmetic Operators
a = np.array([10,20,30])
b = np.array([1,2,3]) 
print(a+b) # [11 22 33]
print(np.add(a,b))
print(a-b) # [ 9 18 27]
print(np.subtract(a,b))
print(a*b) # [10 40 90]
print(np.multiply(a,b))
print(a/b) # [10. 10. 10.] -> division generally produces floating-point values
print(np.divide(a,b))
print(a//b) # [10 10 10] -> floor division
print(np.floor_divide(a,b))
print(a%b) # [0 0 0]
print(np.mod(a,b))
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr % 2 == 0) # [False  True False  True False  True]
print(a**2) # power -> each value is squared
print(np.power(a,2))


# example - temperature conversion
centigrade = np.array([0,10,20,30])
fahrenheit = (centigrade*9/5)+32
print(fahrenheit) # [32. 50. 68. 86.]

# example - salary transformation
salary = np.array([10000,12500,40000])
print(salary*1.01) # [10100. 12625. 40400.]



# Mathematical Functions 

# 1. np.abs()
actual = np.array([100, 200, 300])
predicted = np.array([90, 220, 280])
error = np.abs(actual - predicted)
print(error) # [10 20 20]


# 2. np.square()
arr = np.array([1, 2, 3, 4])
print(np.square(arr))
 
# 3. np.exp() -> Calculates: e^x
arr = np.array([0, 1, 2])
print(np.exp(arr)) # [1. 2.71828183 7.3890561 ]

# 4. np.log() -> Natural logarithm:
arr = np.array([1, 2, 10])
print(np.log(arr)) 


# 5. np.log10() -> Base-10 logarithm:
arr = np.array([1, 10, 100, 1000])
print(np.log10(arr)) #  [0. 1. 2. 3.]


"""
Log transformations are frequently used in:

Data analysis
Statistics
Machine Learning
"""

# 6. Rounding
arr = np.array([1.234, 2.567, 3.891])
print(np.round(arr)) # [1. 3. 4.]

# Specify decimals:
print(np.round(arr, 2)) # [1.23 2.57 3.89]

# 7. np.floor() -> Always rounds downward:
arr = np.array([1.2, 2.8, -1.5])
print(np.floor(arr)) # [1 2 -2]

# 8. np.ceil() -> Always rounds upward:
print(np.ceil(arr)) # [ 2 3 -1]


"""
NumPy Aggregation Functions
Aggregation = combining many values into one result.

arr = np.array([10, 20, 30, 40, 50])
Function	Purpose	Example	Result
np.sum(arr) equivalent to arr.sum() => 150
np.prod(arr)   => 24
np.min(arr)    => 10
np.max(arr)    => 50
np.mean(arr)   => 30.0
np.median(arr) => 30.0
np.std(arr)	   => (spread)
np.var(arr)	   => np.var(arr) == np.std(arr) ** 2 (spread**2)
"""



# The Axis Concept - important concept in all of numpy

# 1. no axis
print(np.array([[1,2,3,4,5],[6,7,8,9,10]]).sum()) # returns sum of all elements # 55
arr = np.array([[1,2,3,4,5],[6,7,8,9,10]]) 
print(np.sum(arr,axis=0)) # 1+6 2+7 3+8 4+9 5+10 = [7 9 11 13 15] 
print(np.sum(arr,axis=1)) # [1+2+3+4+5,6+7+8+9+10] 


# axis = 0 => collapses the rows and sums the columns   => move vertically -> operation down each column
# axis = 1 => collapses the columns and sums each rows  => move horizontally -> operation down each row


arr = np.array([
    [10, 20, 30],
    [40, 50, 60]
])
print(np.sum(arr)) # 1+2+3+4+5+6+7+8+9 = 45
print(np.sum(arr,axis=0)) # [12 15 18] -> 1+4+7,2+5+8,3+6+9 (sum of col 0, sum of col 1, sum of col 2)
print(np.sum(arr,axis=1)) # [6 15 24] -> 1+2+3,4+5+6,7+8+9 (sum of row 0, sum of row 1, sum of row 2)


arr = np.array([
    [10, 20, 30],
    [40, 50, 60]
])
# Column sums
np.sum(arr, axis=0) # [50 70 90]
# Row sums
np.sum(arr, axis=1) # [60 150]



# axis with mean - this calculates the mean of each feature/column
print(np.mean(arr,axis=0))
x = np.array([
    [20, 50000],
    [25, 60000],
    [30, 80000]
])

# Each column is a feature: => Column 0 → Age ||Column 1 → Salary
# Calculate the average of each feature:

print(np.mean(x, axis=0)) # [25. 63333.33333333]
# This is exactly the type of calculation used in data preprocessing

arr = np.array([[10,50,30],[20,40,60]])
print(np.min(arr,axis=0)) # [10 40 30] -> gives column minimums
print(np.min(arr,axis=1)) # [10 20] -> gives row minimums
print(np.max(arr,axis=0)) # [20 50 60] -> gives column maximums
print(np.max(arr,axis=1)) # [50 60] -> gives row maximum


# keepdims = True
arr = np.array([
    [10, 20, 30],
    [40, 50, 60]
])
result=np.sum(arr,axis=0)
print(result.shape) # (3,) -> shape of the result is (3,) instead of (1,3) -> keeps the dimensions of the original array
result=np.sum(arr,axis=0,keepdims=True) # [[50 70 90]] -> shape : (1,3) instead of (3,) -> keeps the dimensions of the original array
print(result.shape) # (1,3) -> shape of the result is (1,3) instead of (3,) -> keeps the dimensions of the original array
result = np.sum(arr,axis=1,keepdims=True) 
print(result) # [[ 60] [150]] -> shape : (2,1) instead of (2,) -> keeps the dimensions of the original array
print(result.shape) # (2,1) -> shape of the result is (2,1) instead of (2,) -> keeps the dimensions of the original array



# cumulative operations - calculates progressively from the beginning
arr=np.array([1,2,3,4])
print(np.cumsum(arr)) 


# example : running total example
sales = np.array([100,230,440,560])
running_sales = np.cumsum(sales)
print(running_sales)

arr = np.array([1,2,3,4,5])
product=np.cumprod(arr)
print(product) # [1 2 6 24 120] -> 1, 1*2, 1*2*3, 1*2*3*4, 1*2*3*4*5

arr = np.array([[1,2,3],[4,5,6]])
diff=np.diff(arr)
print(diff) # [[1 1] [1 1]] -> calculates the difference between each element and its predecessor along the specified axis. By default, it operates along the last axis (axis=-1). In this case, it computes the difference between adjacent elements in each row.

"""
useful for time series analysis
change detection
sequential data
"""


# comparison operations - imp before boolean masking
# with numpy arrays use &,| not and / or
arr = np.array([10,20,30,40,50])
print(arr>30)
print(arr<30)
print(arr==30)
print(arr!=30)
print(arr[(arr>10)&(arr<40)])
print(arr[(arr>10)&(arr<40)])


# np.where(condition,value if true, value if false)
a = np.array([10,20,30])
result = np.where(arr>25,1,0)
print(result) # if value is > 25 then 1 else 0 -> [0 0 1 1 1]

# without true of false
indices = np.where(arr>25) 
print(result) # it returns indices where the condition is true -> (array([2, 3, 4]),) -> indices of elements greater than 25

# example - exam pass / fail
marks = np.array([35, 72, 48, 90, 28])
result = np.where(marks >= 40, "Pass", "Fail")
print(result) # ['Fail' 'Pass' 'Pass' 'Pass' 'Fail']


# element wise multiplication vs matrix multiplication
a = np.array([1,2,3])
b = np.array([4,5,6])
print(a*b) # [4 10 18]
print(np.dot(a,b)) # dot product of a & b : 32


# matrix element wise multiplication
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8,9],[10,11,12]])
print(a*b) # element wise multiplication -> # [[ 7 16 27] [40 55 72]]
c = np.array([[1,2],[3,4]]) 
d = np.array([[5,6],[7,8]])
print(c @ d) # matrix multiplication -> [[19 22] [43 50]]

# a*b -> matrix element wise multiplication
# a @ b -> matrix multiplication


# compare array element by element
# useful for clipping values, data preprocessing, numerical algorithms
a = np.array([1,6,7])
b = np.array([2,10,3])
print(np.maximum(a,b)) # [2 10 7]
print(np.minimum(a,b)) # [1 6 3]


# np.clip() - useful for handling outliers and limiting numerical ranges, data preprcessing
arr = np.array([5,10,20,30,40])
print(np.clip(arr,10,30)) # values below 10 becomes 10 above 30 becomes 30 # [10 10 20 30 30]




# np.nan and Aggregation

arr = np.array([10, 20, np.nan, 40])
# np.mean(arr) returns: nan, Because the array contains a missing value.
print(np.nanmean(arr))  # Output: 23.33333333

# Similarly:

np.nansum(arr)
np.nanmin(arr)
np.nanmax(arr)
np.nanstd(arr)

"""
These ignore NaN values.
You will study missing values in greater detail later.
"""



# out Parameter - This is an advanced concept.

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.add(a, b) # -> NumPy creates a new array.

# You can provide an output array:
result = np.empty(3, dtype=int)
np.add(a, b, out=result)
print(result) # Output: [5 7 9]

"""
This can reduce unnecessary memory allocation.
For your current roadmap:
Understand that out exists, but you do not need to use it regularly yet.
"""




# Data Type and Arithmetic -> Data types affect calculations.

arr = np.array([1, 2, 3], dtype=np.int32)
#Multiplication:
result = arr * 2
# NumPy handles the operation according to the data types involved.
arr = np.array([1, 2, 3])
print(arr / 2) # Output: [0.5 1.  1.5]



"""
Division usually produces floating-point values.
Always be careful when:
Converting floats to integers
Working with very large numbers
Using different data types
PART O: Complete ML Example
Suppose we have:

X = np.array([
    [20, 50000],
    [25, 60000],
    [30, 80000],
    [35, 90000]
])

Columns:

Column 0 → Age
Column 1 → Salary
Find column means
mean = np.mean(X, axis=0)

print(mean)

Output:

[2.75e+01 7.00e+04]

Approximately:

Age average = 27.5
Salary average = 70000
Find column minimums
np.min(X, axis=0)

Output:
[20 50000]
Find column maximums
np.max(X, axis=0)

Output:
[35 90000]
Center the data
X_centered = X - np.mean(X, axis=0)

This subtracts each feature's average from its values.
This is a basic idea behind feature preprocessing.
"""



# Important Difference Between sum() and np.sum()

"""
np.sum(arr) & arr.sum()
For normal use, they are effectively equivalent.
You will commonly see both styles. """

arr = np.array([1, 2, 3])
print(np.sum(arr))
print(arr.sum())

# Both produce: 6

"""
The same applies to:

arr.mean()
arr.min()
arr.max()
arr.std()
"""


"""
Phase 4 Practice Problems -> You should solve these yourself.

Beginner Level
Problem 1

Given:
arr = np.array([10, 20, 30, 40, 50])
Perform:
Add 5
Subtract 10
Multiply by 2
Divide by 5
Find the square of every value

Problem 2
Given:
a = np.array([10, 20, 30])
b = np.array([2, 4, 5])
Calculate:
a + b # [12 24 35]
a - b # [8 16 25]
a * b # [20 80 50]
a / b # [5. 5. 6.]
a // b #[5 5 6]
a % b # [0 0 0]

Problem 3
Find:
sum        # np.sum(arr)
product    # np.prod(arr)
minimum    # np.min(arr)
maximum    # np.max(arr)
mean       # np.mean(arr)
median     # np.median(arr)
standard deviation   # np.std(arr)
variance   # np.var(arr)

for:
arr = np.array([10, 20, 30, 40, 50])



Intermediate Level

Problem 4
Given:
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

Find:
1. Total sum → adds all elements
print(np.sum(arr))

# 2. Sum of each column → axis=0 means column-wise
print(np.sum(arr, axis=0))

# 3. Sum of each row → axis=1 means row-wise
print(np.sum(arr, axis=1))

# 4. Mean of each column → calculate average column-wise
print(np.mean(arr, axis=0))

# 5. Maximum of each row → find maximum value row-wise
print(np.max(arr, axis=1))

# 6. Minimum of each column → find minimum value column-wise
print(np.min(arr, axis=0))mum of each row



Problem 5
Given:
marks = np.array([35, 42, 78, 29, 90, 55])
Using np.where():
Convert marks ≥ 40 to "Pass" # result = np.where(marks >= 40, "Pass", "Fail")
Convert marks < 40 to "Fail"


Problem 6
Given:
arr = np.array([10, 20, 30, 40, 50])

# Greater than 25
print(arr[arr > 25])

# Less than 45
print(arr[arr < 45])

# Between 20 and 50
print(arr[(arr >= 20) & (arr <= 50)])

# Even values
print(arr[arr % 2 == 0])


Problem 7
Given:
sales = np.array([100, 250, 150, 300, 200])
Calculate:
1. Running sales total # print(np.cumsum(sales))
2. Difference between consecutive sales # print(np.diff(sales))
3. Total sales # print(np.sum(sales))
4. Average sales # print(np.mean(sales))


Advanced Level

Problem 8
Given:
X = np.array([
    [20, 50000, 3],
    [25, 60000, 5],
    [30, 80000, 7],
    [35, 90000, 10]
])

Calculate:
Mean of each feature                  # mean = np.mean(X, axis=0)
Minimum of each feature               # np.min(X, axis=0)
Maximum of each feature               # np.max(X, axis=0)
Standard deviation of each feature    # np.std(X, axis=0)
Center the dataset by subtracting the feature means           # centered_X = X - mean


Problem 9
Given:
A = np.array([
    [1, 2],
    [3, 4]
])
B = np.array([
    [5, 6],
    [7, 8]
])
Find:
A*B # Element-wise multiplication
A @ B # matrix multiplication
np.dot(A, B) # scalar dot product
Then explain why the results are different.


Problem 10
Given:
arr = np.array([
    [10, 20, 30],
    [40, 50, 60]
])
Calculate:
np.sum(arr, axis=1)             # result = np.sum(arr, axis=1)
Then calculate it with:
keepdims=True                   # result_keep = np.sum(arr, axis=1, keepdims=True)
Compare the shapes.
Phase 4 Challenge
Write a program that:
data = np.array([
    [20, 50000],
    [25, 60000],
    [30, 80000],
    [35, 90000],
    [40, 100000]
])
Performs the following:
Calculate the mean of each column.                                 # mean = np.mean(data, axis=0)
Calculate the minimum of each column.                              # minimum = np.min(data, axis=0)
Calculate the maximum of each column.                              # maximum = np.max(data, axis=0)
Center the data by subtracting the column means.                   # centered = data - mean
Calculate the absolute difference from the mean.                   # absolute_difference = np.abs(data - mean)
Find all values greater than the column mean.                      # greater_than_mean = data > mean
Clip all values to a selected range.                               # clipped = np.clip(data, [20, 50000], [40, 90000])
Calculate the cumulative sum of the data.                          # cumulative_sum = np.cumsum(data, axis=0)
Calculate the standard deviation of each column.                   # std = np.std(data, axis=0)
This single exercise combines many important NumPy concepts.
"""

