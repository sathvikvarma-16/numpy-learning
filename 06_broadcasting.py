# Broadcasting
"""
Broadcasting is one of the most important NumPy concepts for Machine Learning.
The simple idea is:
Broadcasting allows NumPy to perform operations between arrays of different shapes without manually copying values.
"""

# 1. Simple example
import numpy as np
arr = np.array([1, 2, 3])
print(arr + 10) # Output: [11 12 13]

# What happened? You can mentally imagine NumPy doing:

"""
[1, 2, 3]
   +10
 ↓
[11, 12, 13]

NumPy effectively applies 10 to every element.
You don't have to write: arr + np.array([10, 10, 10])

That's broadcasting.
"""

# 2. Matrix + Vector


# Consider:
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
vector = np.array([10, 20, 30])

# Shapes:

# matrix.shape : (2, 3) & vector.shape : (3,)
# matrix + vector -> Output:

"""
[[11 22 33]
 [14 25 36]]

Visualize it
The vector:
[10 20 30]
is applied to each row:

[1  2  3]       [10 20 30]
[4  5  6]   +   [10 20 30]

Giving:

[11 22 33]
[14 25 36]

You didn't actually create a second copy of the vector. NumPy handles this through broadcasting.

3. Why is this called broadcasting?

Imagine:

Matrix:

1  2  3
4  5  6

Vector:

10 20 30

NumPy conceptually "broadcasts" the vector across the rows:

10 20 30
10 20 30

Then performs:

1  2  3   +   10 20 30
4  5  6   +   10 20 30
-----------------------
11 22 33
14 25 36

Again, conceptually duplicated. NumPy does not necessarily physically copy the data like this.

4. Shape compatibility
This is the most important part.
NumPy compares array dimensions from right to left.

For example:
Matrix shape:  (2, 3)
Vector shape:     (3,)

Align them from the right:
(2, 3)
   (3)

Think of (3,) as:

(1, 3)

Now:
(2, 3)
(1, 3)

Compare:
3 = 3  ✓
2 vs 1 ✓

Therefore broadcasting works.
"""




# 5. The two important compatibility rules

"""
Two dimensions are compatible when:

Rule 1: They are equal
3 and 3 ✓
5 and 5 ✓
Rule 2: One of them is 1
3 and 1 ✓
5 and 1 ✓

Otherwise:
3 and 4 ✗
2 and 5 ✗

So:
(2, 3)
(1, 3)

✅ Compatible.

But:
(2, 3)
(2,)

Let's align:
(2, 3)
   (2)

Equivalent:
(2, 3)
(1, 2)

Compare:
3 vs 2 ✗
So this doesn't broadcast.
"""


# 6. A very important example
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

x = np.array([10, 20])

# Shapes: matrix → (2, 3) & x → (2,)

"""
Align:
(2, 3)
(1, 2)

Compare from right:
3 vs 2 ❌

Therefore:
matrix + x
❌ Error.

This is why shape understanding is essential.
"""


# 7. Broadcasting with a column vector

x = np.array([
    [10],
    [20]
])

# Shape: (2, 1)
# Matrix: (2, 3)

"""
Compare:
(2, 3)
(2, 1)

From right:

3 vs 1 ✓
2 vs 2 ✓

So broadcasting works:
matrix + x

Conceptually:

1  2  3       10 10 10
4  5  6   +   20 20 20
-------------------------
11 12 13
24 25 26

Output:

[[11 12 13]
 [24 25 26]]
"""



# 8. Broadcasting is not only addition -> It works with many operations.

# Subtraction
arr = np.array([10, 20, 30])
print(arr - 5) # [ 5 15 25]
print(arr * 2) # multiplication  -> [20 40 60]
print(arr / 10) # division -> [1. 2. 3.]



# 9. Broadcasting + ML

"""
This is where you need to understand why you're learning it.
Suppose you have a dataset:

       Age  Salary  Experience
Person1 20   30000      1
Person2 25   40000      3
Person3 30   50000      5

Represent it as: """

X = np.array([
    [20, 30000, 1],
    [25, 40000, 3],
    [30, 50000, 5]
])

"""
Suppose you want to subtract the average of every feature.
First:  """
mean = np.mean(X, axis=0)
"""
You get one value for each column.
Then:
X_centered = X - mean
Here, mean is broadcast across every row.
This kind of operation appears constantly in ML preprocessing, normalization, neural-network calculations, and tensor operations.

🧠 The mental model you should remember
Scalar + array
[1 2 3]
+ 10
↓
[11 12 13]

Scalar is broadcast everywhere.

Matrix + row vector
1 2 3       10 20 30
4 5 6   +   10 20 30

Vector is broadcast across rows.
Matrix + column vector
1 2 3       10 10 10
4 5 6   +   20 20 20

Column vector is broadcast across columns.
🔥 Your Phase 6 checklist
You do not need to study every broadcasting edge case right now.
For your current AI/ML roadmap, master these:
✅ What broadcasting means
✅ Scalar + array
✅ Matrix + vector
✅ (2,3) with (3,)
✅ (2,3) with (2,1)
✅ Shape compatibility
✅ Compare dimensions from right → left
✅ Dimensions are compatible when they are equal or one is 1
✅ Understand why incompatible shapes produce an error
✅ Understand broadcasting in ML preprocessing

One rule to burn into memory:
Broadcasting compares shapes from right to left. Each pair must either be equal or one of them must be 1.
Once this is clear, you're ready to move into the next NumPy topics that matter for ML.

"""