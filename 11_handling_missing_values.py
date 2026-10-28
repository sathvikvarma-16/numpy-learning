# ============================================================
## PHASE 11: HANDLING MISSING AND SPECIAL VALUES IN NUMPY
# ============================================================

import numpy as np

"""
Real-world data is often messy.
A dataset may contain:

Missing values
NaN values
Infinite values
Invalid calculations
Empty or unavailable numerical data


NumPy provides several tools to identify and handle these values.
Main concepts:

np.nan
np.inf
np.isnan()
np.isinf()
np.nan_to_num()


The goal of this phase is NOT to become an expert in data cleaning.
The goal is to understand:

1. What NaN means
2. What infinity means
3. How to detect them
4. How to replace them
5. How these concepts are useful in ML preprocessing

"""

# ============================================================
# 1. WHAT IS np.nan?
# ============================================================

"""
NaN means: Not a Number
It represents a missing or undefined numerical value.

Example:
A person's salary is missing.
A sensor failed to record a value.
A calculation produced an undefined result.
NumPy represents this using: np.nan
"""

arr = np.array([1, np.nan, 3])
print(arr) # [ 1. nan  3.]

"""
The array contains:
1
missing value
"""


# ============================================================
# 2. CHECK FOR NaN USING np.isnan()
# ============================================================

arr = np.array([1, np.nan, 3])
result = np.isnan(arr)
print(result)
"""
Output:
[False  True False]
Explanation:
1   -> not NaN -> False
NaN -> NaN     -> True
3   -> not NaN -> False

np.isnan() checks EVERY element individually.
"""


# ============================================================
# 3. BOOLEAN MASKING WITH np.isnan()
# ============================================================

arr = np.array([10, np.nan, 30, np.nan, 50])
mask = np.isnan(arr)
print(mask)

"""
Output:
[False  True False  True False]
"""


# We can use this Boolean array as a mask.
print(arr[mask])  # this gives the elements that are nan
"""
Output:
[nan nan]
"""


# ============================================================
# 4. FIND VALUES THAT ARE NOT NaN
# ============================================================


arr = np.array([10, np.nan, 30, np.nan, 50])
mask = ~np.isnan(arr)

print(mask)
"""
~ means NOT.
np.isnan(arr):
[False True False True False]
~np.isnan(arr):
[True False True False True]
"""

print(arr[mask]) # [10. 30. 50.] -> These are the values that are NOT NaN.



# ============================================================
# 5. IMPORTANT: NaN IS SPECIAL
# =====================================================s=======

"""
One strange thing about NaN:
np.nan == np.nan
does NOT return True.

NaN is not considered equal to itself.
So do NOT use:
arr == np.nan
to find NaN values.

Instead ALWAYS use:
np.isnan(arr)
"""

print(np.nan == np.nan) # False

arr = np.array([1, np.nan, 3])
print(arr == np.nan) # [False False False]

"""
This is why:
np.isnan()
is the correct method.
"""



# ============================================================
# 6. WHAT IS np.inf?
# ============================================================


"""
np.inf means: positive infinity
It represents a value larger than any finite numerical value.

Example:
"""

print(np.inf) # inf

arr = np.array([10, 20, np.inf])
print(arr) # [10 20 inf]



# ============================================================
# 7. NEGATIVE INFINITY
# ============================================================


"""
NumPy also supports negative infinity: -np.inf
"""


arr = np.array([
10,
20,
np.inf,
-np.inf
])
print(arr) # [10 20 inf -inf]



# ============================================================
# 8. HOW DOES INFINITY APPEAR?
# ============================================================


"""
Infinity can appear from mathematical operations.
For example: 1/0


in pure Python raises an error.
But NumPy operations can produce infinity in floating-point
calculations.
"""

arr = np.array([1.0, 2.0, 0.0])
result = np.divide(1, arr) # divide 1 with array elements
print(result) # [1.  0.5  inf]
"""

Because:
1 / 1 = 1
1 / 2 = 0.5
1 / 0 = infinity in this NumPy operation
"""



# ============================================================
# 9. np.isinf()
# ============================================================



"""
np.isinf() checks whether values are positive or negative infinity.
"""

arr = np.array([
10,
np.inf,
20,
-np.inf,
30
])

result = np.isinf(arr)
print(result) # [False  True False  True False] => Only the infinite values are marked True.



# ============================================================
# 10. SELECT INFINITE VALUES
# ============================================================

arr = np.array([
10,
np.inf,
20,
-np.inf,
30
])

print(arr[np.isinf(arr)]) # [ inf -inf]



# ============================================================
# 11. NaN VS INFINITY
# ============================================================


"""
These are DIFFERENT
np.nan:
Not a Number
Usually represents missing or undefined numerical data.

np.inf: +ve infinity

-np.inf: -ve infinity

Example: [10, np.nan, 20, np.inf, -np.inf]

"""


# ============================================================
# 12. CHECK BOTH NaN AND INFINITY
# ============================================================

arr = np.array([
10,
np.nan,
20,
np.inf,
-np.inf,
30
])

print(np.isnan(arr)) # [False  True False False False False] -> NaN is detected by np.isnan().
print(np.isinf(arr)) # [False False False  True  True False] -> Infinity is detected by np.isinf().



# ============================================================
# 13. CHECK FOR NaN OR INFINITY TOGETHER
# ============================================================

arr = np.array([
10,
np.nan,
20,
np.inf,
-np.inf,
30
])
mask = np.isnan(arr) | np.isinf(arr)
print(mask)  # this finds both nan, +inf/-inf
print(arr[mask])   # [nan inf -inf]



# ============================================================
# 14. FIND VALID FINITE VALUES
# ============================================================


"""
A finite value is:

```
NOT NaN
AND
NOT infinity
```
NumPy provides: np.isfinite()
This is extremely useful.
"""

arr = np.array([
10,
np.nan,
20,
np.inf,
-np.inf,
30
])

print(np.isfinite(arr))  # [ True False True False False True] -> Only normal finite numerical values are True.



# ============================================================
# 15. FILTER ONLY FINITE VALUES
# ============================================================

arr = np.array([
10,
np.nan,
20,
np.inf,
-np.inf,
30
])

finite_values = arr[np.isfinite(arr)]
print(finite_values)  # [10. 20. 30.] -> This removes: NaN, +inf, -inf from the selected result.



# ============================================================
# 16. np.nan_to_num()
# ============================================================


"""
np.nan_to_num() is used to replace: NaN / +infinity / -infinity with numerical values.

Basic example:
"""

arr = np.array([
1,
np.nan,
3
])
result = np.nan_to_num(arr)
print(result) # [1. 0. 3.] ->  by default Nan is 0



# ============================================================
# 17. REPLACE NaN WITH A CUSTOM VALUE
# ============================================================

arr = np.array([
10,
np.nan,
30
])
result = np.nan_to_num(arr, nan=100)
print(result) # [ 10. 100.  30.] => here NaN=100



# ============================================================
# 18. REPLACE INFINITY
# ============================================================

arr = np.array([
10,
np.inf,
20,
-np.inf,
30
])


result = np.nan_to_num(
arr,
posinf=999,
neginf=-999
)
print(result) # [  10.  999.   20. -999.   30.] i.e +inf -> 999, -inf -> -999



# ============================================================
# 19. REPLACE NaN AND INFINITY TOGETHER
# ============================================================



arr = np.array([
10,
np.nan,
20,
np.inf,
-np.inf,
30
])
result = np.nan_to_num(
arr,
nan=0,
posinf=100,
neginf=-100
)

print(result)  # [  10.    0.   20.  100. -100.   30.]



# ============================================================
# 20. WHY NaN CAN CAUSE PROBLEMS
# ============================================================

"""
NaN can spread through calculations.
Example:
[10, NaN, 30]
If we calculate the mean normally:
np.mean()
the result can become NaN.
"""

arr = np.array([10, np.nan, 30])
print(np.mean(arr)) # output : nan

"""
Why?
Because the NaN value participates in the calculation.
"""



# ============================================================
# 21. np.nanmean()
# ============================================================

"""
NumPy provides special functions that IGNORE NaN.
For mean: np.nanmean()
"""

arr = np.array([10, np.nan, 30])
print(np.nanmean(arr)) # 20 => ignores nan and calculates mean of 10 and 30



# ============================================================
# 22. OTHER NaN-AWARE FUNCTIONS
# ============================================================


"""
There are NaN-aware versions of several statistical functions.
Examples:

np.nanmean()
np.nanmedian()
np.nanstd()
np.nanvar()
np.nansum()
np.nanmin()
np.nanmax()

These functions generally ignore NaN values.
"""

arr = np.array([
10,
np.nan,
20,
30
])

print(np.nanmean(arr))
print(np.nanmedian(arr))
print(np.nanstd(arr))
print(np.nanvar(arr))
print(np.nansum(arr))
print(np.nanmin(arr))
print(np.nanmax(arr))

"""
These calculations ignore the NaN value.
"""




# ============================================================
# 23. np.sum() VS np.nansum()
# ============================================================

arr = np.array([
10,
np.nan,
30
])

print(np.sum(arr))

print(np.nansum(arr)) # 40 => Because nansum() ignores NaN.



# ============================================================
# 24. np.min() VS np.nanmin()
# ============================================================


arr = np.array([
10,
np.nan,
30
])
print(np.min(arr))
print(np.nanmin(arr)) 



# ============================================================
# 25. REPLACING NaN WITH THE MEAN
# ============================================================

"""
This is a very important DATA PREPROCESSING idea.

Suppose:
[10, 20, NaN, 40, 50]

We could replace the missing value with the mean
of the available values.
First calculate the mean while ignoring NaN.
"""

arr = np.array([
10,
20,
np.nan,
40,
50
])

mean = np.nanmean(arr)
print(mean) # 30
arr[np.isnan(arr)] = mean
print(arr)  # [10. 20. 30. 40. 50.] => The missing value was replaced with the mean.




# ============================================================
# 26. REPLACING NaN WITH THE MEDIAN
# ============================================================

arr = np.array([
10,
20,
np.nan,
40,
50
])

median = np.nanmedian(arr)
arr[np.isnan(arr)] = median
print(arr) # [10 20 30 40 50]

"""
The missing value was replaced with the median.

In real ML preprocessing, whether you use mean,
median, another value, or remove the row depends
on the dataset and the problem.
"""



# ============================================================
# 27. PRACTICAL DATA CLEANING EXAMPLE
# ============================================================

data = np.array([
10,
20,
np.nan,
40,
np.inf,
60
])

print(data) # [10, 20, NaN, 40, inf, 60]


# Check missing values.
print(np.isnan(data))
# Check infinite values.
print(np.isinf(data))
# Check valid finite values.
print(np.isfinite(data))

"""
This gives us three different ways of understanding
the condition of our data.
"""



# ============================================================
# 28. CLEANING SPECIAL VALUES
# ============================================================

data = np.array([
10,
20,
np.nan,
40,
np.inf,
60
])

cleaned = np.nan_to_num(
data,
nan=0,
posinf=0,
neginf=0
)

print(cleaned) # [10. 20.  0. 40.  0. 60.] => NaN  -> 0 / +inf -> 0 / -inf -> 0


"""
BUT:
Do not blindly replace every missing value with 0
in a real ML project.
The correct strategy depends on what the missing value means.
"""


# ============================================================
# 29. CHECK HOW MANY NaN VALUES EXIST
# ============================================================

arr = np.array([
10,
np.nan,
20,
np.nan,
30,
np.nan
])

count_nan = np.sum(np.isnan(arr))
print(count_nan) # 3

"""
Explanation:

np.isnan(arr) => [False, True, False, True, False, True]
In NumPy: False = 0 / True = 1
So: 0 + 1 + 0 + 1 + 0 + 1 = 3
"""



# ============================================================
# 30. COUNT INFINITE VALUES
# ============================================================

arr = np.array([
10,
np.inf,
20,
-np.inf,
30
])

count_inf = np.sum(np.isinf(arr))
print(count_inf) # 2

"""

There are two infinite values: +inf, -inf
"""


# ============================================================
# 31. COUNT ALL INVALID VALUES
# ============================================================


arr = np.array([
10,
np.nan,
20,
np.inf,
-np.inf,
30
])
invalid = np.isnan(arr) | np.isinf(arr)
count_invalid = np.sum(invalid)
print(count_invalid) # 3



# ============================================================
# 32. 2D ARRAY WITH MISSING VALUES
# ============================================================


data = np.array([
[10, 20, 30],
[40, np.nan, 60],
[70, 80, 90]
])
print(np.isnan(data))

"""
output :
[[False False False]
 [False  True False]
 [False False False]
]

The result has the SAME SHAPE as the original array.
Original shape: (3,3)
Boolean mask: (3,3)
Every position tells us whether that position contains NaN.
"""


# ============================================================
# 33. FIND NaN POSITIONS
# ============================================================


data = np.array([
[10, 20, 30],
[40, np.nan, 60],
[70, 80, 90]
])

positions = np.argwhere(np.isnan(data))
print(positions) # [[1 1]] => this means row=1 and column=1


"""
contains NaN.
Remember:
NumPy uses zero-based indexing.
So:
row 1
column 1
is the center value.
"""


# ============================================================
# 34. REPLACE NaN IN A 2D ARRAY
# ============================================================


data = np.array([
[10, 20, 30],
[40, np.nan, 60],
[70, 80, 90]
])
data[np.isnan(data)] = 0
print(data)


"""
Output:
[[10. 20. 30.]
[40.  0. 60.]
[70. 80. 90.]]

Every NaN was replaced with 0.
"""



# ============================================================
# 35. REPLACE NaN USING COLUMN MEAN
# ============================================================


"""
This is a more realistic preprocessing idea.
Suppose:

Feature 1   Feature 2   Feature 3

   10          20          30
   40          NaN         60
   70          80          90

Instead of using one overall mean,
we can calculate the mean for each column.
For this particular example:
Feature 1 mean:
(10 + 40 + 70) / 3 = 40

Feature 2 mean:
(20 + 80) / 2 = 50

Feature 3 mean:
(30 + 60 + 90) / 3 = 60

The missing Feature 2 value can therefore be replaced by 50.
"""

data = np.array([
[10, 20, 30],
[40, np.nan, 60],
[70, 80, 90]
])
column_means = np.nanmean(data, axis=0) # 
print(column_means) # [40 50 60]



# Find the NaN positions.
rows, columns = np.where(np.isnan(data)) # Returns coordinate index arrays (rows, columns) indicating exactly where missing values live. For this matrix, it targets row 1 and column 1

# Replace each missing value with its column mean.
data[rows, columns] = column_means[columns] # : Uses advanced integer array indexing to extract the calculated mean for column 1 (which is 50.0) and mutates the original layout in place.
print(data)

"""
Output:

[[10. 20. 30.]
[40. 50. 60.]
[70. 80. 90.]]

This introduces an important ML concept : missing-value imputation
We are filling missing values using information
from the available data.
"""



# ============================================================
# 36. IMPORTANT: DON'T ALWAYS REMOVE NaN
# ============================================================

"""
When you find missing data, there are several possible strategies:

1. Remove the row
2. Remove the column
3. Replace with mean
4. Replace with median
5. Replace with a constant
6. Use a more advanced imputation method

There is NO single correct method for every dataset.
For now, understand the NumPy operations.
You will study proper data-cleaning strategies more deeply
when you learn Pandas and ML preprocessing.
"""



# ============================================================
# 37. SPECIAL VALUES CAN COME FROM CALCULATIONS
# ============================================================

"""
NaN and infinity are not always explicitly present
in the original dataset.
They can appear because of mathematical operations.

Examples include:
```
division by zero
invalid mathematical operations
overflow
missing values propagating through calculations
```
"""



# ============================================================
# 38. NaN CAN PROPAGATE
# ============================================================

arr = np.array([10, np.nan, 30])
print(arr + 10) # [20. nan 40.] => The NaN remains NaN.



# ============================================================
# 39. INFINITY CAN ALSO PARTICIPATE IN CALCULATIONS
# ============================================================


arr = np.array([10, np.inf, 30]) # [20. inf 40.]

print(arr + 10)



# ============================================================
# 40. CHECKING A DATASET BEFORE ML
# ============================================================


data = np.array([
[10, 20, 30],
[40, np.nan, 60],
[70, np.inf, 90]
])

"""
Before using numerical data for ML, it is useful to check: this kind of checking is a data preprocessing

Are there NaN values?
Are there infinite values?
Are all values finite?
"""

print("NaN values:")
print(np.isnan(data))

print()

print("Infinite values:")
print(np.isinf(data))

print()

print("Finite values:")
print(np.isfinite(data))




# ============================================================
# 41. np.isfinite() AS A DATA VALIDATION TOOL
# ============================================================


data = np.array([
10,
20,
np.nan,
40,
np.inf
])

if np.all(np.isfinite(data)):
    print("All values are valid and finite.")
else:
    print("Dataset contains NaN or infinity.")



"""
np.all() checks whether ALL values are True.
If even one value is:
NaN
+inf
-inf

np.isfinite() will contain False.
"""

# ============================================================
# 42. IMPORTANT DIFFERENCE
# ============================================================

"""
MEMORIZE THESE:
np.isnan(arr)
    Checks for NaN.

np.isinf(arr)
    Checks for positive or negative infinity.

np.isfinite(arr)
    Checks for normal finite values.

np.nan_to_num(arr)
    Replaces NaN and infinity with numerical values.
"""



# ============================================================
# 43. QUICK COMPARISON
# ============================================================


"""
Suppose:
arr = [10, NaN, inf, -inf, 20]
np.isnan(arr) # [False, True, False, False, False]
np.isinf(arr) # [False, False, True, True, False]
np.isfinite(arr) # [True, False, False, False, True]
"""



# ============================================================
# 44. IMPORTANT NaN-AWARE STATISTICS
# ============================================================

"""
Normal statistics may be affected by NaN.
Use NaN-aware functions when appropriate:

```
np.nanmean()
np.nanmedian()
np.nanstd()
np.nanvar()
np.nansum()
np.nanmin()
np.nanmax()
```

Example:
"""

data = np.array([
10,
20,
np.nan,
40,
50
])

print("Mean:", np.nanmean(data))
print("Median:", np.nanmedian(data))
print("Std:", np.nanstd(data))
print("Variance:", np.nanvar(data))
print("Sum:", np.nansum(data))
print("Minimum:", np.nanmin(data))
print("Maximum:", np.nanmax(data))



# ============================================================
# 45. COMPLETE DATA CLEANING EXAMPLE
# ============================================================

data = np.array([
10,
20,
np.nan,
40,
np.inf,
60
])

"""
Step 1:
Check the dataset.
"""

print("Original:")
print(data)

"""
Step 2:
Check NaN.
"""

print("NaN mask:")
print(np.isnan(data))

"""
Step 3:
Check infinity.
"""

print("Infinity mask:")
print(np.isinf(data))

"""
Step 4:
Check whether everything is finite.
"""

print("Finite mask:")
print(np.isfinite(data))

"""
Step 5:
Replace special values if that is appropriate
for the preprocessing strategy.
"""

cleaned = np.nan_to_num(
data,
nan=0,
posinf=0,
neginf=0
)

print("Cleaned:")
print(cleaned)


# ============================================================
# 46. MACHINE LEARNING CONNECTION
# ============================================================


"""
Imagine a dataset:
Age    Salary    Experience

20     30000     1
25     NaN       3
30     50000     5
The missing salary is represented as NaN.
Before feeding the data into an ML model,
we usually need to decide how to handle that missing value.
Possible strategy:

Find the median salary
Replace the missing salary with that value
NumPy gives us the tools to:

detect the NaN
calculate statistics
replace the NaN

Later, Pandas and scikit-learn will provide
more powerful tools for this process.
"""


# ============================================================
# 47. VERY IMPORTANT ML WARNING
# ============================================================

"""
Do NOT blindly do this:
np.nan_to_num(data)
and assume the dataset is ready for ML.
Why?
Because:
NaN -> 0
may not make sense for your data.
For example:
Salary = NaN
does NOT necessarily mean:
Salary = 0

The correct treatment depends on what the missing value means.
For now:
Learn detection.
Learn replacement.
Understand the consequences.
Later, learn proper imputation strategies.
"""


# ============================================================
# 48. FINAL QUICK REFERENCE
# ============================================================

"""
np.nan
Represents Not a Number / missing or undefined value.

np.inf
Positive infinity.

-np.inf
Negative infinity.

np.isnan(arr)
Finds NaN values.

np.isinf(arr)
Finds positive and negative infinity.

np.isfinite(arr)
Finds values that are neither NaN nor infinity.

np.nan_to_num(arr)
Replaces NaN and infinity with numerical values.

np.nanmean(arr)
Mean while ignoring NaN.

np.nanmedian(arr)
Median while ignoring NaN.

np.nanstd(arr)
Standard deviation while ignoring NaN.

np.nanvar(arr)
Variance while ignoring NaN.

np.nansum(arr)
Sum while ignoring NaN.

np.nanmin(arr)
Minimum while ignoring NaN.

np.nanmax(arr)
Maximum while ignoring NaN.
"""


# ============================================================
# 49. FINAL MEMORY MAP
# ============================================================


"""
SPECIAL VALUES
|
-----------------------------
|                           |
NaN                       Infinity
|                           |
np.nan                    np.inf
|                           |
np.isnan()                np.isinf()
|                           |
----------- CHECK -----------
|
np.isfinite()
|
CLEAN / HANDLE
|
np.nan_to_num()
|
ML DATA PREPROCESSING
"""



# ============================================================
# 50. PRACTICE QUESTIONS
# ============================================================

"""
DO THESE YOURSELF.

---

## QUESTION 1

Create:

arr = np.array([10, np.nan, 20, 30])
Find which values are NaN.
"""

"""
## QUESTION 2
Using the same array, select only the non-NaN values.
"""


"""
## QUESTION 3
arr = np.array([10, np.inf, 20, -np.inf, 30])
Find all infinite values.
"""


"""
## QUESTION 4
Check whether all values in this array are finite:
arr = np.array([10, 20, np.nan])
"""

"""
## QUESTION 5
Replace NaN with 0 using np.nan_to_num().
"""

"""
## QUESTION 6
Replace:
NaN  -> 100
+inf -> 999
-inf -> -999
"""

"""
## QUESTION 7
Find the mean while ignoring NaN:
arr = np.array([10, 20, np.nan, 40])
"""

"""
## QUESTION 8
Count how many NaN values are present:

arr = np.array([
10,
np.nan,
20,
np.nan,
30
])
"""

"""

## QUESTION 9
Count how many infinite values are present:

arr = np.array([
10,
np.inf,
20,
-np.inf,
30
])
"""


"""
## QUESTION 10
Given:

data = np.array([
[10, 20],
[30, np.nan],
[50, 60]
])

Replace the NaN with 0.
"""


# ============================================================
# PHASE 11 COMPLETE
# ============================================================

"""
1. np.nan
   Represents Not a Number.
   Commonly used for missing or undefined numerical data.

2. np.inf
   Represents positive infinity.

3. -np.inf
   Represents negative infinity.

4. np.isnan(arr)
   Detects NaN.

5. np.isinf(arr)
   Detects positive and negative infinity.

6. np.isfinite(arr)
   Detects normal finite values.

7. np.nan_to_num(arr)
   Replaces NaN and infinity with numerical values.

8. np.nanmean(arr)
   Calculates mean while ignoring NaN.

9. np.nanmedian(arr)
   Calculates median while ignoring NaN.

10. NaN-aware functions
    np.nanstd()
    np.nanvar()
    np.nansum()
    np.nanmin()
    np.nanmax()

MOST IMPORTANT MEMORY:

NaN       -> np.isnan()
Infinity  -> np.isinf()
Valid     -> np.isfinite()
Replace   -> np.nan_to_num()

ML CONNECTION:

Real data
    ↓
Missing / invalid values
    ↓
Detect them
    ↓
Decide how to handle them
    ↓
Clean data
    ↓
ML preprocessing
    ↓
Model


At this stage, you do NOT need advanced imputation theory.
You need to be completely comfortable with:

np.nan
np.inf
np.isnan()
np.isinf()
np.isfinite()
np.nan_to_num()

and understand WHY missing and special values
must be handled before many ML workflows.
"""

