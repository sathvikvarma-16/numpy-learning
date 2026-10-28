# ============================================================
# PHASE 10: COMBINING AND SPLITTING ARRAYS IN NUMPY
# ============================================================

import numpy as np

"""
PHASE 10 GOAL
-------------

In this phase, we will learn how to:

1. Combine arrays
2. Understand np.concatenate()
3. Understand np.vstack()
4. Understand np.hstack()
5. Understand np.stack()
6. Understand the difference between concatenate() and stack()
7. Split arrays using np.split()
8. Work with 1D arrays
9. Work with 2D arrays
10. Understand axis while combining arrays
11. Understand how this connects to Machine Learning

Main functions:

```
np.concatenate()
np.vstack()
np.hstack()
np.stack()
np.split()
```

IMPORTANT:

Do not just memorize the functions.

You need to understand:

```
"Where is the new data being added?"
"Are we increasing rows?"
"Are we increasing columns?"
"Are we creating a new dimension?"
```

"""


# ============================================================
# 1. np.concatenate()
# ============================================================


"""
np.concatenate() joins arrays together along an existing axis.
Syntax:
```
np.concatenate((array1, array2))
```
For 1D arrays, this is very simple.
"""

a = np.array([1, 2])
b = np.array([3, 4])

result = np.concatenate((a, b)) # concatenation of two 1d arrays gives a 1d array
print(result)

"""
Output:
[1 2 3 4]

The two arrays were joined together.
"""


# ============================================================
# 2. MORE concatenate() EXAMPLES
# ============================================================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.concatenate((a, b)))

"""
Output:
[1 2 3 4 5 6]
"""


# ============================================================
# 3. CONCATENATING THREE ARRAYS
# ============================================================

a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])

result = np.concatenate((a, b, c))
print(result)

"""
Output:
[1 2 3 4 5 6]

You can concatenate more than two arrays.
"""


# ============================================================
# 4. concatenate() WITH 2D ARRAYS
# ============================================================

a = np.array([
[1, 2],
[3, 4]
])
b = np.array([
[5, 6],
[7, 8]
])
# concatenation of two 2 d arrays can be done using axis parameter. By default axis=0 to concatenate along rows otherwise axis=1 can be used to concatenate along columns
"""
a:
1 2
3 4

b:
5 6
7 8
"""


# ============================================================
# 5. concatenate() axis=0
# ============================================================

result = np.concatenate((a, b), axis=0)
print(result)

"""
Output:

[[1 2]
[3 4]
[5 6]
[7 8]]

axis=0 means we are joining DOWNWARD.
Think:
```
a
↓
b
```
Original:
1 2
3 4

5 6
7 8

Result:
1 2
3 4
5 6
7 8

The number of ROWS increased.
Shape:
a -> (2, 2)
b -> (2, 2)

result -> (4, 2)
"""

# ============================================================
# 6. concatenate() axis=1
# ============================================================

result = np.concatenate((a, b), axis=1)
print(result)
"""
Output:

[[1 2 5 6]
[3 4 7 8]]

axis=1 means joining SIDE BY SIDE.

Think:

```
a | b
```
Original:
1 2    5 6
3 4    7 8

Result:
1 2 5 6
3 4 7 8

The number of COLUMNS increased.
Shape:

a -> (2, 2)
b -> (2, 2)

result -> (2, 4)
"""


# ============================================================
# 7. IMPORTANT concatenate() RULE
# ============================================================


"""
When concatenating 2D arrays:
axis=0:
```
Rows are added.
```
axis=1:
```
Columns are added.
```


Remember:
```
axis=0 -> DOWN
axis=1 -> ACROSS
```

For concatenate():
The dimensions OTHER than the chosen axis must match.
Example:
```
(2, 3)
(2, 3)
```

axis=0:
```
2 + 2 = 4
```

Result:
```
(4, 3)
```

axis=1:
```
3 + 3 = 6
```

Result:
```
(2, 6)
```


"""
# ============================================================
# 8. np.vstack()
# ============================================================

"""
vstack means: Vertical Stack
It stacks arrays vertically.
Think:
```
one array
    ↓
another array
```
So it usually increases the NUMBER OF ROWS.
"""

a = np.array([
[1, 2],
[3, 4]
])

b = np.array([
[5, 6],
[7, 8]
])

result = np.vstack((a, b))
print(result)

"""
Output:
[[1 2]
[3 4]
[5 6]
[7 8]]

Same basic result as:
np.concatenate((a, b), axis=0)
"""

# ============================================================
# 9. vstack() WITH 1D ARRAYS
# ============================================================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.vstack((a, b))

print(result)

"""
Output:

[[1 2 3]
[4 5 6]]

Notice something important.
Originally:
a.shape -> (3,)
b.shape -> (3,)

After vstack:
result.shape -> (2, 3)
The 1D arrays became rows.
"""


# ============================================================
# 10. np.hstack()
# ============================================================

"""
hstack means : Horizontal Stack

It combines arrays horizontally.
Think:

```
array A | array B
```
It usually increases the NUMBER OF COLUMNS
when working with 2D arrays.
"""


a = np.array([
[1, 2],
[3, 4]
])

b = np.array([
[5, 6],
[7, 8]
])

result = np.hstack((a, b))
print(result)

"""
Output:
[[1 2 5 6]
[3 4 7 8]]

Same basic result as:
```
np.concatenate((a, b), axis=1)
```
"""


# ============================================================
# 11. hstack() WITH 1D ARRAYS
# ============================================================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.hstack((a, b))
print(result)

"""
Output:
[1 2 3 4 5 6]
With 1D arrays, hstack() simply joins them.
Shape:
a -> (3,)
b -> (3,)
result -> (6,)
"""


# ============================================================
# 12. vstack() VS hstack()
# ===========================================================

"""
For 2D arrays:
vstack() : adds rows.

hstack(): add columns.

Example:
a:
1 2
3 4

b:
5 6
7 8

vstack:
1 2
3 4
5 6
7 8

Shape:
(4, 2)

hstack:
1 2 5 6
3 4 7 8

Shape:
(2, 4)
"""


# ============================================================
# 13. np.stack()
# ============================================================

"""
np.stack() is DIFFERENT from concatenate().
This is extremely important.
concatenate(): joins arrays along an EXISTING axis that is => already present in the arrays.

stack(): joins them along a NEW axis that is => not present in the arrays.
In simple words:

```
concatenate -> existing dimension
stack       -> NEW dimension
```


"""
# ============================================================
# 14. Basic stack()
# ============================================================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.stack((a, b))
print(result)

"""
Output:

[[1 2 3]
[4 5 6]]

The original arrays:
a -> shape (3,)
b -> shape (3,)

After stack():
result -> shape (2, 3)
A new dimension was created.

Think of it as:
Two separate arrays:

[1 2 3]

[4 5 6]

became a 2D array:

[1 2 3]
[4 5 6]
"""



# ============================================================
# 15. stack() axis=0
# ============================================================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.stack((a, b), axis=0)
print(result)

"""
Output:
[[1 2 3]
[4 5 6]]

Shape:
(2, 3)
axis=0 creates the new dimension at position 0.
"""


# ============================================================
# 16. stack() axis=1
# ============================================================

result = np.stack((a, b), axis=1)
print(result)

"""
Output:
[[1 4]
[2 5]
[3 6]]

Shape:
(3, 2)

Think:
a: 1 2 3
b: 4 5 6

They are stacked as columns:
1 4
2 5
3 6
"""


# ============================================================
# 17. concatenate() VS stack()
# ============================================================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("concatenate:")
print(np.concatenate((a, b)))

print("stack:")
print(np.stack((a, b)))

"""
concatenate():

[1 2 3 4 5 6]

Shape:
(6,)

It joined them along the existing 1D axis.

stack():

[[1 2 3]
[4 5 6]]

Shape:

(2, 3)

It created a NEW dimension.
This is the most important difference:
```
concatenate -> joins existing dimension
stack       -> creates new dimension
```
"""


# ============================================================
# 18. SHAPE COMPARISON
# ============================================================
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a.shape)
print(b.shape)
print(np.concatenate((a, b)).shape)
print(np.stack((a, b)).shape)
"""
Output:
(3,)
(3,)
(6,)
(2, 3)

So:
concatenate:
```
(3,) + (3,) -> (6,)
```

stack:
```
(3,) + (3,) -> (2, 3)
```

This shape difference is very important.
"""


# ============================================================
# 19. np.split()
# ============================================================

"""
np.split() does the opposite kind of operation.
Instead of combining arrays:
```
concatenate
vstack
hstack
stack
```


we can split an array into smaller arrays.
Syntax:
```
np.split(array, number_of_sections)
```
"""


# ============================================================
# 20. BASIC split()
# ============================================================

arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, 2)
print(result)

"""
Output:
[array([1, 2, 3]), array([4, 5, 6])]

The original array:
[1 2 3 4 5 6]
was split into 2 equal parts:
[1 2 3]
[4 5 6]
"""



# ============================================================
# 21. SPLIT INTO 3 PARTS
# ============================================================

arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, 3)
print(result)

"""
Output:
[array([1, 2]), array([3, 4]), array([5, 6])]
The array had 6 elements.
6 / 3 = 2 elements per section.
"""


# ============================================================
# 22. STORE SPLIT ARRAYS SEPARATELY
# ============================================================


arr = np.array([1, 2, 3, 4, 5, 6])
a, b = np.split(arr, 2)
print(a)
print(b)
"""
Output:
[1 2 3]
[4 5 6]
You can unpack the result into variables.
"""


# ============================================================
# 23. np.split() REQUIRES EQUAL PARTS
# ============================================================

arr = np.array([1, 2, 3, 4, 5])

"""
This will NOT work:
```
np.split(arr, 2)
```
because 5 elements cannot be evenly divided into 2 equal sections.
NumPy will raise an error.
For unequal splitting, use:
```
np.array_split()
```
"""

# ============================================================
# 24. np.array_split()
# ============================================================

arr = np.array([1, 2, 3, 4, 5])
result = np.array_split(arr, 2)
print(result)

"""
array_split() allows unequal-sized sections.
Possible result:
[1 2 3]
[4 5]

Remember:
```
np.split()
    -> equal-sized sections required
np.array_split()
    -> unequal sections are allowed
```
"""


# ============================================================
# 25. SPLITTING 2D ARRAYS
# ============================================================
matrix = np.array([
[1, 2],
[3, 4],
[5, 6],
[7, 8]
])
print(matrix)
"""
Matrix:

1 2
3 4
5 6
7 8

Shape:
(4, 2)
"""

# ============================================================
# 26. split() axis=0
# ============================================================
result = np.split(matrix, 2, axis=0)
print(result)

"""
Output:
[array([
[1, 2],
[3, 4]
]),

array([
[5, 6],
[7, 8]
])]

axis=0 means split DOWN the rows.

So:
First part:
1 2
3 4

Second part:
5 6
7 8
"""


# ============================================================
# 27. split() axis=1
# ============================================================

matrix = np.array([
[1, 2, 3, 4],
[5, 6, 7, 8]
])
result = np.split(matrix, 2, axis=1)
print(result)

"""
Output:
First part:
1 2
5 6

Second part:
3 4
7 8

axis=1 splits the COLUMNS.
"""


# ============================================================
# 28. PRACTICAL EXAMPLE: DATASET ROWS
# ============================================================

data = np.array([
[1, 10],
[2, 20],
[3, 30],
[4, 40],
[5, 50],
[6, 60]
])

"""
Imagine this is a small dataset.
Column 1:
Feature
Column 2:
Target
We might want to divide the dataset into parts.
"""
part1, part2 = np.split(data, 2, axis=0)
print(part1)
print(part2)

"""
The rows are divided into two groups.
"""



# ============================================================
# 29. PRACTICAL ML CONNECTION: BATCHES
# ============================================================


"""
Machine Learning often processes data in BATCHES.
Suppose we have 6 samples:
sample 1
sample 2
sample 3
sample 4
sample 5
sample 6
We can split them into batches.

Example:

Batch 1:
samples 1, 2, 3

Batch 2:
samples 4, 5, 6

This basic idea of splitting data into groups
is important when you later learn ML training.
"""


# ============================================================
# 30. COMBINING DATASET ROWS
# ============================================================

data1 = np.array([
[1, 10],
[2, 20]
])

data2 = np.array([
[3, 30],
[4, 40]
])

combined = np.vstack((data1, data2))
print(combined)

"""
Output:

[[ 1 10]
[ 2 20]
[ 3 30]
[ 4 40]]

We added more ROWS.
This is useful when combining data containing
the same features.
"""


# ============================================================
# 31. COMBINING FEATURES / COLUMNS
# ============================================================

age = np.array([
[20],
[25],
[30]
])

salary = np.array([
[30000],
[40000],
[50000]
])

combined = np.hstack((age, salary))
print(combined)

"""
Output:
[[20 30000]
[25 40000]
[30 50000]]

We added a new COLUMN.
This is useful when combining different features
that belong to the same samples.
"""


# ============================================================
# 32. vstack VS hstack IN ML
# ============================================================



"""
Suppose:
data1:
Age   Salary
20    30000
25    40000

data2:
Age   Salary
30    50000
35    60000

Use vstack() when you want:
MORE SAMPLES / ROWS
```
data1
data2
```

Use hstack() when you want:
MORE FEATURES / COLUMNS
```
Feature A | Feature B
```

This distinction is extremely important in ML.
"""



# ============================================================
# 33. STACKING SAMPLES
# ============================================================


sample1 = np.array([10, 20, 30])
sample2 = np.array([40, 50, 60])
data = np.stack((sample1, sample2))
print(data)

"""
Output:
[[10 20 30]
[40 50 60]]

Each 1D array can represent one sample.
After stack():
```
rows = samples
columns = features
```

Shape:
(2, 3)
2 samples
3 features
"""


# ============================================================
# 34. STACKING MULTIPLE SAMPLES
# ============================================================

sample1 = np.array([10, 20, 30])
sample2 = np.array([40, 50, 60])
sample3 = np.array([70, 80, 90])
data = np.stack((sample1, sample2, sample3))

print(data)

"""
Output:
[[10 20 30]
[40 50 60]
[70 80 90]]

Shape:
(3, 3)

3 samples
3 features
"""


# ============================================================
# 35. concatenate VS vstack VS hstack VS stack
# ============================================================

"""
These functions can look confusing.
Use this mental model:
---

## FUNCTION             MAIN IDEA

np.concatenate() -> Join arrays along an EXISTING axis.

np.vstack()
Stack vertically.
Usually adds ROWS.

np.hstack()
Stack horizontally.
Usually adds COLUMNS.

np.stack()
Create a NEW axis.
------------------

Simple memory:

```
concatenate -> JOIN

vstack      -> ↓ DOWN

hstack      -> → SIDE BY SIDE

stack       -> NEW DIMENSION
```
"""


# ============================================================
# 36. SHAPE THINKING
# ============================================================


"""
Always check shape before combining arrays.
Use:

```
arr.shape
```

Example:
a.shape
b.shape
Then decide which operation makes sense.

For example:
a.shape = (2, 3)
b.shape = (4, 3)


vstack can work:
```
(2, 3)
(4, 3)
```

becomes:
```
(6, 3)
```


because the number of columns matches.
But hstack would not work here because:
```
2 != 4
```
for the row dimension.
"""


# ============================================================
# 37. concatenate() SHAPE EXAMPLE
# ============================================================

a = np.zeros((2, 3))
b = np.ones((4, 3))
print(a.shape)
print(b.shape)

result = np.concatenate((a, b), axis=0)
print(result.shape)

"""
Output:

(2, 3)
(4, 3)
(6, 3)

We joined along axis=0.
Rows:

```
2 + 4 = 6
```

Columns stay:

```
3
```

"""
# ============================================================
# 38. CONCATENATING COLUMNS
# ============================================================

a = np.zeros((3, 2))
b = np.ones((3, 4))
result = np.concatenate((a, b), axis=1)
print(result.shape)

"""
Output:
(3, 6)

Columns:
```
2 + 4 = 6
```

Rows remain:
```
3
```
"""


# ============================================================
# 39. stack() WITH 2D ARRAYS
# ============================================================

a = np.array([
[1, 2],
[3, 4]
])

b = np.array([
[5, 6],
[7, 8]
])
result = np.stack((a, b))
print(result)
print(result.shape)
"""
Output:

[
[
[1, 2],
[3, 4]
],

```
[
    [5, 6],
    [7, 8]
]
```

]

Shape:
(2, 2, 2)

Why?
Original:

a -> (2, 2)
b -> (2, 2)

stack() creates a NEW dimension.
Therefore:

```
(2, 2)
    +
(2, 2)
```
becomes: (2,2,2) to know what is the new dimension ? The new dimension is the first dimension.
This is the beginning of thinking in
higher-dimensional arrays/tensors.
"""



# ============================================================
# 40. stack() axis=0 VS axis=1
# ============================================================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("axis=0")
print(np.stack((a, b), axis=0))
print()
print("axis=1")
print(np.stack((a, b), axis=1))
"""
axis=0:

[[1 2 3]
[4 5 6]]

Shape:
(2, 3)

axis=1:
[[1 4]
[2 5]
[3 6]]

Shape:
(3, 2)

Both create a NEW dimension.
The axis determines where that new dimension is placed.
"""




# ============================================================
# 41. PRACTICAL ML EXAMPLE: COMBINING TRAINING DATA
# ============================================================


"""
Imagine we already have some training samples.
training1:
3 samples
2 features
"""

training1 = np.array([
[10, 100],
[20, 200],
[30, 300]
])
training2 = np.array([
[40, 400],
[50, 500]
])



"""
training1 shape:
(3, 2)
training2 shape:
(2, 2)


Both have 2 features.
So we can add the new samples using vstack().
"""

combined_training = np.vstack((training1, training2))
print(combined_training)
print(combined_training.shape)

"""
Output:
[[ 10 100]
[ 20 200]
[ 30 300]
[ 40 400]
[ 50 500]]

Shape:
(5, 2)

We now have:
5 samples
2 features
"""


# ============================================================
# 42. PRACTICAL ML EXAMPLE: ADDING A FEATURE
# ============================================================

age = np.array([
[20],
[25],
[30],
[35]
])

experience = np.array([
[1],
[3],
[5],
[7]
])


"""
Both arrays contain information about the SAME 4 people.

age:
20
25
30
35

experience:
1
3
5
7

We want:
Age | Experience
So use hstack().
"""

data = np.hstack((age, experience))
print(data) 

"""
Output:
[[20  1]
[25  3]
[30  5]
[35  7]]
"""


# ============================================================
# 43. WHEN TO USE EACH FUNCTION
# ============================================================

"""
Use np.concatenate() when:
You want explicit control over an existing axis.

Example:
np.concatenate((a, b), axis=0)
or:
np.concatenate((a, b), axis=1)
Use np.vstack() when : You want to add rows.

Example:
np.vstack((a, b))

Use np.hstack() when : You want to add columns.
Example:

np.hstack((a, b))
Use np.stack() when:

You want to create a NEW dimension.

Example:
np.stack((a, b))
Use np.split() when:
You want to divide an array into equal parts.

Example:
np.split(arr, 2)
Use np.array_split() when:
The parts do not have to be equal.
"""


# ============================================================
# 44. FINAL MEMORY MAP
# ============================================================


"""
NUMPY ARRAY
|
-------------------------------
|              |              |
COMBINE         SPLIT          SORT
|
----------------
|      |       |
concatenate vstack hstack
|
stack
(NEW DIMENSION)
COMBINE
```
concatenate -> existing axis
vstack      -> add rows
hstack      -> add columns
stack       -> new dimension
```

SPLIT:
```
split       -> equal sections
array_split -> unequal sections allowed
```
"""



# ============================================================
# 45. FINAL QUICK REFERENCE
# ============================================================

"""

1. JOIN 1D arrays:
   np.concatenate((a, b))

2. JOIN 2D arrays vertically:
   np.vstack((a, b))

3. JOIN 2D arrays horizontally:
   np.hstack((a, b))

4. Create a new dimension:
   np.stack((a, b))

5. Split into equal parts:
   np.split(arr, 2)

6. Split into unequal parts if necessary:
   np.array_split(arr, 2)

7. Add rows:
   np.concatenate((a, b), axis=0)
   OR
   np.vstack((a, b))

8. Add columns:
   np.concatenate((a, b), axis=1)
   OR
   np.hstack((a, b))
   """


# ============================================================
# 46. PRACTICE QUESTIONS
# ============================================================


"""
Try these WITHOUT looking at previous examples.
---

## QUESTION 1
Create:
a = [1, 2, 3]
b = [4, 5, 6]
Join them using concatenate().
"""

## QUESTION 2

# Create:

a = [1, 2, 3]
b = [4, 5, 6]
# Use vstack().
# What will the shape be?

## QUESTION 3
# Create:
a = [1, 2, 3]
b = [4, 5, 6]

# Use hstack(). 
# What will the shape be?


## QUESTION 4
"""
Use stack() with axis=0.
What is the resulting shape?
"""


## QUESTION 5

"""
Use stack() with axis=1.
What is the resulting shape?
"""


## QUESTION 6

"""
Create:
arr = [1, 2, 3, 4, 5, 6]
Split it into 3 equal parts.
"""

## QUESTION 7
"""
Create:
data1 = [
[1, 2],
[3, 4]
]

data2 = [
[5, 6],
[7, 8]
]

Use vstack().
What is the resulting shape?
"""

## QUESTION 8

"""
# Using the same data, use hstack().
What is the resulting shape?
"""

## QUESTION 9
"""
Explain the difference between:

```
np.concatenate()
np.stack()
```
in your own words.
"""

## QUESTION 10
"""
You have:
age.shape = (100, 1)
salary.shape = (100, 1)
You want to create a dataset containing:
Age | Salary
Which function should you use?
Think before answering.
"""


# ============================================================
# PHASE 10 COMPLETE
# ============================================================


"""
WHAT YOU MUST REMEMBER
----------------------

```
                concatenate()
                     |
              existing axis
                     |
          ---------------------
          |                   |
       axis=0              axis=1
          |                   |
      add rows            add columns
```
vstack()
-> vertical
-> add rows

hstack()
-> horizontal
-> add columns

stack()
-> creates a NEW dimension

split()
-> divides an array
-> equal sections required

array_split()
-> divides an array
-> unequal sections allowed

MOST IMPORTANT DIFFERENCE:
```
concatenate()
    JOIN along an EXISTING dimension

stack()
    CREATE a NEW dimension
```

ML CONNECTION:

```
vstack()
    -> add more samples

hstack()
    -> add more features

stack()
    -> create an additional dimension/batch structure

split()
    -> divide data into groups/batches
```

If you understand SHAPES while using these functions,
you have understood Phase 10 properly.
"""