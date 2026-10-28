# ============================================================
# PHASE 13: LINEAR ALGEBRA IN NUMPY
# ============================================================

import numpy as np

"""
PHASE 13 GOAL
=============
Linear algebra is one of the most important mathematical
foundations of Machine Learning.

In this phase, you will learn the NumPy operations you
actually need for your AI/ML roadmap.
You need to understand:

1. Vectors
2. Matrices
3. Dot product
4. Matrix multiplication
5. Transpose
6. Identity matrix
7. Norms
8. Solving linear equations
9. Inverse
10. Determinant
11. Eigenvalues
12. Eigenvectors
13. Basic matrix decomposition concepts
14. Linear algebra in Machine Learning
You DO NOT need advanced linear algebra yet.
You should understand what the operations mean,
what shapes are required, and when they are useful.
"""


# ============================================================
# 1. VECTORS
# ============================================================

"""
A vector is a one-dimensional collection of numbers.
Example:
[1, 2, 3]
In NumPy:
"""
v = np.array([1, 2, 3])
print(v)
print(v.shape)

"""
Output:
[1 2 3]
(3,)

The shape (3,) means:
3 elements
1D array

For ML, a vector can represent things such as:
features of one sample
model parameters
coordinates
weights
"""


# ============================================================
# 2. MATRICES
# ============================================================

"""
A matrix is a two-dimensional array.
Example:

1 2 3
4 5 6


In NumPy:
"""
A = np.array([
[1, 2, 3],
[4, 5, 6]
])

print(A)
print(A.shape)

"""
Shape:
(2, 3)

Meaning:
2 rows
3 columns

In Machine Learning, a dataset is commonly represented
as a matrix:
rows    -> samples
columns -> features
"""



# ============================================================
# 3. VECTOR ADDITION
# ============================================================


"""
Vectors with the same shape can be added element by element.
"""

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)

"""
Output:
[5 7 9]

Calculation:
[1+4, 2+5, 3+6]
= [5, 7, 9]
"""


# ============================================================
# 4. SCALAR MULTIPLICATION
# ============================================================


"""
A scalar is a single number.
A vector or matrix can be multiplied by a scalar.
"""

a = np.array([1, 2, 3])
print(3 * a)

"""
Output:
[3 6 9]
"""


# ============================================================
# 5. DOT PRODUCT
# ============================================================


"""
The dot product combines two vectors into ONE number.
For:
a = [1, 2, 3]
b = [4, 5, 6]


the dot product is:
1*4 + 2*5 + 3*6
= 4 + 10 + 18
= 32
"""

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dot(a, b)
print(result)

"""
Output:
32
"""


# ============================================================
# 6. DOT PRODUCT USING @
# ============================================================

"""
For two 1D vectors:
a @ b
also calculates their dot product.
"""
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a @ b)  # 32



# ============================================================
# 7. DOT PRODUCT: ML CONNECTION
# ============================================================


"""
Suppose:
features = [2, 3, 4] & weights = [0.5, 1.0, 2.0]

A simple model can calculate:
prediction = features dot weights

"""
features = np.array([2, 3, 4])
weights = np.array([0.5, 1.0, 2.0])
prediction = features @ weights
print(prediction)


"""
Calculation:
2*0.5 + 3*1.0 + 4*2.0
= 1 + 3 + 8
= 12

This basic operation is at the heart of many ML models.
"""



# ============================================================
# 8. MATRIX MULTIPLICATION
# ============================================================


"""
Matrix multiplication is DIFFERENT from element-wise
multiplication.
Suppose:
A =
1 2
3 4

B =
5 6
7 8

Then:
A @ B

performs matrix multiplication.
The result is:
1*5 + 2*7     1*6 + 2*8
3*5 + 4*7     3*6 + 4*8
"""

A = np.array([
[1, 2],
[3, 4]
])

B = np.array([
[5, 6],
[7, 8]
])

C = A @ B
print(C)

"""
Output:

[[19 22]
[43 50]]
"""


# ============================================================
# 9. HOW MATRIX MULTIPLICATION WORKS
# ============================================================

"""
The rule is:
rows of A
    ×
columns of B

For:

A shape = (m, n)
B shape = (n, p)

the multiplication is valid.
Result : (m, p)


The INNER dimensions must match.
Example:
(2, 3) @ (3, 4) is valid.

Result:
(2, 4)
"""


A = np.random.default_rng(42).random((2, 3))
B = np.random.default_rng(42).random((3, 4))
C = A @ B
print(C.shape)

"""
Output: (2,4)
"""

  
# ============================================================
# 10. MATRIX MULTIPLICATION VISUAL RULE
# ============================================================

"""
Remember this:
(2, 3) @ (3, 4)
   ↑       ↑
   match
Result:
(2, 4)

MEMORY:
INNER dimensions must match.
OUTER dimensions become the result.
(m, n) @ (n, p) -> (m, p)


"""
# ============================================================
# 11. np.matmul()
# ============================================================

"""
np.matmul() performs matrix multiplication.
Example:

np.matmul(A, B) is equivalent to A@B

for the normal matrix multiplication cases
you will use here.
"""


A = np.array([
[1, 2],
[3, 4]
])

B = np.array([
[5, 6],
[7, 8]
])

print(np.matmul(A, B))
print(A @ B)



# ============================================================
# 12. np.dot() VS @
# ============================================================

"""
For the basic vector and matrix cases:
np.dot(a, b)
and:
a @ b

can perform the same mathematical operation.
For matrix multiplication, prefer:
A @ B
because it clearly communicates:
MATRIX MULTIPLICATION
You should understand np.dot()
because you will see it frequently in existing code.
"""


# ============================================================
# 13. ELEMENT-WISE MULTIPLICATION
# ============================================================

"""
IMPORTANT DIFFERENCE:
A * B
means:
element-by-element multiplication

while:
A @ B

means:
matrix multiplication
"""


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

"""
A * B:

[[ 5 12]
[21 32]]

A @ B:

[[19 22]
[43 50]]

DO NOT confuse:
* with @
"""



# ============================================================
# 14. TRANSPOSE
# ============================================================

"""
Transpose changes rows into columns
and columns into rows.
Example:
A =

1 2 3
4 5 6

Transpose:
1 4
2 5
3 6
"""


A = np.array([
[1, 2, 3],
[4, 5, 6]
])

print(A.T)

"""
Output:
[[1 4]
[2 5]
[3 6]]
"""



# ============================================================
# 15. np.transpose()
# ============================================================

"""
The same operation can be performed using:
np.transpose()
"""

A = np.array([
[1, 2, 3],
[4, 5, 6]
])

print(np.transpose(A))



# ============================================================
# 16. WHY TRANSPOSE MATTERS IN ML
# ============================================================

"""
Suppose:
X shape = (100, 5)
This means:
100 samples
5 features

Then 
X.T has shape: (5, 100)
Transpose is frequently used in:

matrix calculations
linear regression
neural networks
optimization
statistics
"""


# ============================================================
# 17. IDENTITY MATRIX
# ============================================================

"""
An identity matrix is the matrix equivalent of 1.
For a 3 x 3 identity matrix:
1 0 0
0 1 0
0 0 1

NumPy:
np.eye(3)
"""

I = np.eye(3)
print(I)
"""
Output:

[[1. 0. 0.]
[0. 1. 0.]
[0. 0. 1.]]
"""



# ============================================================
# 18. IDENTITY MATRIX PROPERTY
# ============================================================
"""
For a compatible matrix A:
A @ I = A
and:
I @ A = A
"""

A = np.array([
[1, 2],
[3, 4]
])

I = np.eye(2)
print(A @ I)
print(I @ A)

"""
The result is A in both cases.
"""



# ============================================================
# 19. NORMS
# ============================================================

"""
A norm measures the size or magnitude of a vector.
For example:
[3, 4]
has Euclidean norm:
sqrt(3^2 + 4^2)
= 5
"""


v = np.array([3, 4])
print(np.linalg.norm(v)) # 5.0


# ============================================================
# 20. VECTOR NORM
# ============================================================

"""
The most common norm you should know initially is
the L2 norm, also called Euclidean norm.

For:
[x1, x2, ..., xn]

L2 norm is:
sqrt(x1^2 + x2^2 + ... + xn^2)

NumPy:
np.linalg.norm()
"""

v = np.array([1, 2, 3])
print(np.linalg.norm(v)) # sqrt(14
    

# ============================================================
# 21. L1 NORM
# ============================================================

"""
L1 norm is the sum of absolute values.
For:
[-2, 3, -4]

L1 norm:
|-2| + |3| + |-4|

= 2 + 3 + 4
= 9
"""


v = np.array([-2, 3, -4])
print(np.linalg.norm(v, ord=1)) # 9


  
# ============================================================
# 22. WHY NORMS MATTER IN ML
# ============================================================


"""
Norms are important for:
measuring vector size
distance
regularization
optimization
feature analysis


You will later see:
L1 regularization
L2 regularization

in Machine Learning.
For now, understand:
L1 -> sum of absolute values
L2 -> Euclidean magnitude
"""


# ============================================================
# 23. SOLVING LINEAR EQUATIONS
# ============================================================


"""
Suppose we have:
x + y = 5
2x + y = 8
We can write:
A @ x = b
where:
A = [[1, 1],
     [2, 1]]

x = [x, y]
b = [5, 8]
"""

A = np.array([
[1, 1],
[2, 1]
])

b = np.array([5, 8])
solution = np.linalg.solve(A, b)
print(solution)

"""
Output:
[3. 2.]

Therefore:
x = 3
y = 2
"""


# ============================================================
# 24. np.linalg.solve() IS IMPORTANT
# ============================================================

"""
When you need to solve:
A @ x = b
use:
np.linalg.solve(A, b)
This is generally preferable to explicitly calculating:
inv(A) @ b

when solving a linear system.
So:
solve()
is an important function to know.
"""


# ============================================================
# 25. MATRIX INVERSE
# ============================================================

"""
The inverse of a matrix A is written:
A^-1
The important property is:
A @ A^-1 = I
where I is the identity matrix.
NumPy : np.linalg.inv()
"""


A = np.array([
[1, 2],
[3, 4]
])

A_inv = np.linalg.inv(A)
print(A_inv)


# ============================================================
# 26. VERIFY THE INVERSE
# ============================================================

A = np.array([
[1, 2],
[3, 4]
])

A_inv = np.linalg.inv(A)
print(A @ A_inv)

"""
The result should be approximately:
[[1, 0],
 [0, 1]]

Small floating-point errors may appear.
"""


# ============================================================
# 27. WHEN DOES AN INVERSE EXIST?
# ============================================================

"""
Not every matrix has an inverse.
For a square matrix:
determinant != 0
means the matrix is invertible.

If:
determinant = 0

the matrix is singular
and does not have a normal inverse.
"""


# ============================================================
# 28. DETERMINANT
# ============================================================

"""
The determinant is a special number calculated
from a square matrix.
NumPy:
np.linalg.det()
"""


A = np.array([
[1, 2],
[3, 4]
])

det = np.linalg.det(A)
print(det)

"""
For this matrix:
det = (1*4) - (2*3)
    = 4 - 6
    = -2
"""


# ============================================================
# 29. DETERMINANT AND INVERSE
# ============================================================


"""
Remember:

determinant != 0
    ->
matrix can have an inverse

determinant = 0
    ->
matrix is singular
    ->
inverse does not exist
"""


# ============================================================
# 30. SINGULAR MATRIX
# ============================================================

A = np.array([
[1, 2],
[2, 4]
])

print(np.linalg.det(A))
"""
The determinant is approximately zero.
The second row is simply:
2 * first row
Therefore the matrix does not have a normal inverse.
"""
# Do NOT try np.linalg.inv(A) here.


# ============================================================
# 31. EIGENVALUES AND EIGENVECTORS
# ============================================================

"""
This is the most advanced concept in this phase.
You do NOT need to master the mathematical theory yet.
You need to understand what NumPy gives you.
For a square matrix:
A
an eigenvector v satisfies:
A @ v = lambda * v
where:
lambda = eigenvalue
v      = eigenvector
Meaning:
The matrix transforms the eigenvector without changing
its fundamental direction, only scaling it.
"""


# ============================================================
# 32. np.linalg.eig()
# ============================================================

A = np.array([
[2, 0],
[0, 3]
])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:")
print(eigenvalues)
print()
print("Eigenvectors:")
print(eigenvectors)

"""
For this simple diagonal matrix:
eigenvalues are 2 and 3.

The columns of eigenvectors contain
the corresponding eigenvectors.
"""


# ============================================================
# 33. UNDERSTANDING eig() OUTPUT
# ============================================================

"""
np.linalg.eig(A) returns TWO things:
eigenvalues
eigenvectors

Example:
eigenvalues, eigenvectors = np.linalg.eig(A)
The eigenvectors are stored as COLUMNS.

So:
eigenvectors[:, 0]
is the eigenvector corresponding to:
eigenvalues[0] and eigenvectors[:, 1]
corresponds to : eigenvalues[1]
"""


# ============================================================
# 34. VERIFY AN EIGENPAIR
# ============================================================

A = np.array([
[2, 0],
[0, 3]
])

eigenvalues, eigenvectors = np.linalg.eig(A)
v = eigenvectors[:, 0]
lam = eigenvalues[0]
left = A @ v
right = lam * v
print(left)
print(right)


"""
They should be approximately equal.
This verifies:
A @ v = lambda * v
"""


# ============================================================
# 35. WHY EIGENVALUES MATTER
# ============================================================

"""
Eigenvalues and eigenvectors appear in:
dimensionality reduction
Principal Component Analysis (PCA)
covariance analysis
optimization
scientific computing


You do NOT need to derive eigenvalues by hand now.
You need to understand:
what they are
what eig() returns
how to interpret the result
"""


# ============================================================
# 36. MATRIX RANK
# ============================================================

"""
Rank tells us the number of independent directions
contained in a matrix.
NumPy:
np.linalg.matrix_rank()
"""


A = np.array([
[1, 2],
[2, 4]
])
print(np.linalg.matrix_rank(A))

"""
Output:
1
Why?
The second row is:
2 * first row
So there is only one independent row direction.
"""


# ============================================================
# 37. FULL-RANK MATRIX
# ============================================================

A = np.array([
[1, 2],
[3, 4]
])

print(np.linalg.matrix_rank(A))
"""
Output:
2
Both rows provide independent information.
"""


# ============================================================
# 38. WHY RANK MATTERS
# ============================================================

"""
Rank is useful for understanding:
linear dependence
redundant features
singular matrices
systems of equations

You do not need advanced rank theory yet.
Just understand:
higher rank
    ->
more independent information

lower rank
    ->
more redundancy / dependence
"""



# ============================================================
# 39. DIAGONAL MATRIX
# ============================================================

"""
A diagonal matrix has values only on its main diagonal.
Example:
2 0 0
0 5 0
0 0 7
"""

A = np.diag([2, 5, 7])
print(A)


# ============================================================
# 40. EXTRACT DIAGONAL
# ============================================================

A = np.array([
[2, 0, 0],
[0, 5, 0],
[0, 0, 7]
])
print(np.diag(A))


"""
Output:
[2 5 7]
"""


# ============================================================
# 41. TRACE
# ============================================================

"""
The trace is the sum of the main diagonal.
For:

1 2
3 4

trace:
1 + 4 = 5
"""


A = np.array([
[1, 2],
[3, 4]
])
print(np.trace(A))

"""
Output : 5
"""


# ============================================================
# 42. BASIC MATRIX DECOMPOSITION
# ============================================================

"""
At this stage, you should know that matrices can be
decomposed into simpler matrices.
Important decompositions include:

LU decomposition
QR decomposition
SVD


You do NOT need to derive these manually yet.
You only need to know what they are used for.
Especially important for ML : SVD
"""


# ============================================================
# 43. SINGULAR VALUE DECOMPOSITION
# ============================================================

"""
SVD means : Singular Value Decomposition
A matrix can be decomposed into:
U
S
Vt
using : np.linalg.svd()
Example:
"""

A = np.array([
[1, 2],
[3, 4],
[5, 6]
])

U, S, Vt = np.linalg.svd(A)
print("U:")
print(U)
print()
print("Singular values:")
print(S)
print()
print("Vt:")
print(Vt)
"""
You do NOT need to memorize the derivation now.
Know that SVD is useful for:

```
dimensionality reduction
compression
noise reduction
numerical methods
PCA-related computations
```
"""



# ============================================================
# 44. RECONSTRUCTING A MATRIX USING SVD
# ============================================================


"""
For learning purposes, we can reconstruct A from its
SVD components.
The shapes must be handled correctly.
"""


A = np.array([
[1, 2],
[3, 4],
[5, 6]
])

U, S, Vt = np.linalg.svd(A)
S_matrix = np.zeros_like(A, dtype=float)
S_matrix[:len(S), :len(S)] = np.diag(S)
reconstructed = U @ S_matrix @ Vt
print(reconstructed)

"""
The reconstructed matrix is approximately
the original matrix A.

This demonstrates that SVD decomposes a matrix
into components that can be multiplied back together.
"""


# ============================================================
# 45. CONDITION NUMBER
# ============================================================

"""
The condition number tells us how sensitive a problem
can be to small changes in the input.
NumPy:
np.linalg.cond()
You do not need advanced numerical analysis yet.
Just know:
Very large condition number
    ->
potentially unstable / sensitive problem

This becomes useful when studying numerical
linear algebra more deeply.
"""


A = np.array([
[1, 2],
[3, 4]
])
print(np.linalg.cond(A)) # np.linalg.cond(A) calculates the condition number of a matrix. 
# It tells us how sensitive the solution of a linear system is to small changes in its input data.



# ============================================================
# 46. MATRIX NORMS
# ============================================================

"""
np.linalg.norm() can also calculate norms of matrices.
Example:
"""

A = np.array([
[1, 2],
[3, 4]
])
print(np.linalg.norm(A))

"""
By default, this gives the Frobenius norm for a matrix.
You do not need to memorize every possible matrix norm
at this stage.
"""



# ============================================================
# 47. BASIC LINEAR REGRESSION CONNECTION
# ============================================================


"""
Linear regression can be expressed using matrix operations.
A simplified form is:
y = Xw

where:
X = feature matrix
w = weights
y = prediction

Example:
"""

X = np.array([
[1, 2],
[3, 4],
[5, 6]
])

w = np.array([
2,
3
])

y = X @ w
print(y)


"""
This performs matrix-vector multiplication.

For each sample, the features are combined
with the corresponding weights.

This is one of the most important ML connections
to understand from NumPy linear algebra.
"""



# ============================================================
# 48. MATRIX-VECTOR MULTIPLICATION
# ============================================================

"""
If:
X shape = (3, 2) and w = shape(2,)
then :
X @ w
produces: shape(3,)
Shape rule : (3, 2) @ (2,) -> (3,)
"""

print(X.shape)
print(w.shape)
print(y.shape)



# ============================================================
# 49. BATCH ML CALCULATION
# ============================================================

"""
Suppose:
100 samples
5 features

Then:
X shape = (100, 5)
Weights:
w shape = (5,)

Then:
X @ w
produces : (100,)
That means one prediction for each sample.
"""


rng = np.random.default_rng(42)
X = rng.random((100, 5))
w = rng.random(5)
predictions = X @ w
print(X.shape)
print(w.shape)
print(predictions.shape)

"""
Output:
(100, 5)
(5,)
(100,)
"""


# ============================================================
# 50. NEURAL NETWORK CONNECTION
# ============================================================

"""
A basic neural-network layer also relies heavily on
matrix multiplication.
Conceptually:
output = input @ weights + bias

For example:
input shape:
    (batch_size, input_features)

weights shape:
    (input_features, output_features)

Then : input @ weights
produces:
(batch_size, output_features)
This is why understanding:

vectors
matrices
shapes
transpose
matrix multiplication
is extremely important before learning
neural networks and PyTorch.
"""


# ============================================================
# 51. EXAMPLE NEURAL NETWORK LAYER
# ============================================================

rng = np.random.default_rng(42)
# 4 samples, 3 input features
X = rng.random((4, 3))
# 3 input features, 2 output neurons
W = rng.random((3, 2))
# One bias value for each output neuron
b = rng.random(2)
output = X @ W + b
print("X shape:", X.shape)
print("W shape:", W.shape)
print("b shape:", b.shape)
print("Output shape:", output.shape)

"""
Shapes:
X = (4, 3)
W = (3, 2)
X @ W = (4, 2)
b = (2,)
Output = (4, 2)

This is the basic mathematical structure
behind a fully connected neural-network layer.
"""



# ============================================================
# 52. LINEAR ALGEBRA + BROADCASTING
# ============================================================

"""
Notice : X @ W

produces:
(4, 2)
and:
b
has shape : (2,)
NumPy broadcasts b across all 4 rows.
Therefore:
X @ W + b works.

This connects : Phase-6 Broadcasting
with:
PHASE 13: Linear Algebra
This combination is extremely important in ML.
"""


# ============================================================
# 53. IMPORTANT SHAPE RULES
# ============================================================

"""
MEMORIZE THESE.
Vector : (n,)
Matrix : (rows, columns)
Matrix multiplication:
(m, n) @ (n, p)

Result : (m, p)
Matrix-vector multiplication : (m, n) @ (n,)
Result : (m,)
Transpose:
(m, n).T

Result : (n,m)
"""


# ============================================================
# 54. COMMON ERROR
# ============================================================


"""
This will fail:
A = shape (2, 3)
B = shape (2, 3)
A @ B

Why?
The inner dimensions are:
3 and 2

They do not match.
For matrix multiplication:
(2, 3) @ (3, something)
is required.
"""


# ============================================================
# 55. DOT PRODUCT VS MATRIX MULTIPLICATION
# ============================================================

"""
VECTOR:
a = [1, 2, 3]
b = [4, 5, 6]

a @ b
produces ONE NUMBER.
MATRIX:

A shape = (2, 3)
B shape = (3, 4)

A @ B
produces a:
(2, 4) matrix.

The operation is based on the same
row-by-column multiplication idea.
"""


# ============================================================
# 56. LINEAR ALGEBRA FUNCTIONS YOU SHOULD KNOW
# ============================================================

"""
CORE:
np.dot()
    Dot product.


@
    Matrix multiplication.


np.matmul()
    Matrix multiplication.


A.T
    Transpose.


np.eye()
    Identity matrix.


np.linalg.norm()
    Vector/matrix norm.


np.linalg.solve()
    Solve A @ x = b.


np.linalg.inv()
    Matrix inverse.


np.linalg.det()
    Determinant.


np.linalg.eig()
    Eigenvalues and eigenvectors.


np.linalg.matrix_rank()
    Matrix rank.


np.linalg.svd()
    Singular Value Decomposition.


np.trace()
    Sum of diagonal elements.


np.diag()
    Create/extract diagonal.


np.linalg.cond()
    Condition number.
"""


# ============================================================
# 57. WHAT YOU MUST UNDERSTAND
# ============================================================


"""
DO NOT just memorize functions.
You must understand:
Vector
    ↓
Matrix
    ↓
Shape
    ↓
Dot product
    ↓
Matrix multiplication
    ↓
Transpose
    ↓
Norm
    ↓
Solve equations
    ↓
Inverse
    ↓
Determinant
    ↓
Eigenvalues/eigenvectors
    ↓
SVD
    ↓
Machine Learning
"""



# ============================================================
# 58. WHAT YOU DO NOT NEED YET
# ============================================================


"""
You DO NOT need to study these deeply at this stage:
abstract vector spaces
formal proofs
advanced tensor algebra
advanced eigenvalue theory
Jordan normal form
advanced numerical analysis
advanced matrix calculus
detailed SVD proofs
advanced optimization mathematics
```

Those can come later if your ML specialization requires them.
For your current NumPy stage,
understanding the computational operations and shapes
is the priority.
"""


# ============================================================
# 59. PHASE 13 PRACTICE
# ============================================================

"""
PRACTICE 1
----------

Create:
a = [1, 2, 3]
b = [4, 5, 6]

Calculate their dot product.


PRACTICE 2
-----------
Create:
A = [[1, 2],
     [3, 4]]
Calculate:
A @ A


PRACTICE 3
----------

Create a:
2 x 3
matrix.

Print its transpose.

PRACTICE 4
----------

Create a 3 x 3 identity matrix.



PRACTICE 5
----------
Calculate the Euclidean norm of:
[3,4]


PRACTICE 6
----------

Solve:

x + y = 5
2x + y = 8
using:
np.linalg.solve()


PRACTICE 7
----------

Calculate the inverse of:


[[1, 2],
 [3, 4]]



PRACTICE 8
----------

Calculate the determinant of:

[[1, 2],
 [3, 4]]


PRACTICE 9
----------

Calculate eigenvalues and eigenvectors of:

[[2, 0],
 [0, 3]]



PRACTICE 10
-----------

Create:
X shape = (10, 3)
W shape = (3, 2)

Calculate:
X @ W

What is the resulting shape?
"""



# ============================================================
# 60. FINAL CHECKLIST
# ============================================================

"""
Before leaving Phase 13, you should be able to explain
these WITHOUT looking at your notes:

[ ] What is a vector?

[ ] What is a matrix?

[ ] What does shape (3,) mean?

[ ] What does shape (3, 4) mean?

[ ] What is a dot product?

[ ] What is matrix multiplication?

[ ] Difference between * and @?

[ ] Why must inner dimensions match?

[ ] What does transpose do?

[ ] What is an identity matrix?

[ ] What is a vector norm?

[ ] What does np.linalg.solve() do?

[ ] What is a matrix inverse?

[ ] What does determinant tell us?

[ ] What is a singular matrix?

[ ] What does np.linalg.eig() return?

[ ] What is matrix rank?

[ ] What is SVD used for?

[ ] Why is matrix multiplication important in ML?

[ ] Why are shapes important in neural networks?
"""



# ============================================================
# PHASE 13 COMPLETE
# ============================================================


"""
FINAL MEMORY
============


                LINEAR ALGEBRA
                       |
         ---------------------------
         |                         |
      VECTORS                   MATRICES
         |                         |
    Dot product              Matrix multiplication
         |                         |
         ----------- @ ------------
                       |
                   TRANSPOSE
                       |
                     NORMS
                       |
             LINEAR EQUATIONS
                       |
                   solve()
                       |
             ------------------
             |                |
           inverse         determinant
             |
          inv()
                       |
                EIGENVALUES
                EIGENVECTORS
                       |
                     eig()
                       |
                      SVD
                       |
                     svd()
                       |
                MACHINE LEARNING
                       |
          -------------------------
          |           |           |
         Data      Regression   Neural
      operations                Networks


MOST IMPORTANT ML IDEA:
X @ W + b

This simple expression contains three major
NumPy concepts:

```
X @ W
    -> matrix multiplication

+ b
    -> broadcasting

shapes
    -> determine whether the operation is valid

Understanding this well will make later
ML and PyTorch concepts much easier.
"""
