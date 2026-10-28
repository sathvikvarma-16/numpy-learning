import numpy as np

"""

============================================================
PHASE 14: NUMPY FOR ML
============================================================

This is the FINAL required NumPy phase.

Goal:
Understand how NumPy is used to represent and prepare
data before giving it to a Machine Learning model.

ML FLOW:

Dataset
   ↓
NumPy Array
   ↓
Features X + Labels y
   ↓
Shape Checking
   ↓
Cleaning
   ↓
Transformation
   ↓
Model
"""


# ============================================================
# 1. DATASET
# ============================================================


# Suppose we have:
# Age, Salary -> Features
# Purchased   -> Label

data = np.array([
    [20, 20000, 0],
    [25, 30000, 1],
    [30, 40000, 1]
])



# ============================================================
# 2. FEATURES (X) AND LABELS (y)
# ============================================================

# X = input information given to the model
# y = answer/output the model should learn

X = data[:, :2]
y = data[:, 2]

print("X:")
print(X)

print("y:")
print(y)


# ============================================================
# 3. SAMPLES
# ============================================================

# Each ROW = one sample
# Here:
# 3 rows = 3 samples
# 2 columns = 2 features
print("Number of samples:", X.shape[0])


# ============================================================
# 4. FEATURES
# ============================================================

# Each COLUMN = one feature
# Feature 1 = Age
# Feature 2 = Salary

print("Number of features:", X.shape[1])


# ============================================================
# 5. SHAPE
# ============================================================

print("X shape:", X.shape)   # (3, 2)
print("y shape:", y.shape)   # (3,)

"""
X -> 2D
(3, 2)
3 samples, 2 features

y -> 1D
(3,)
3 labels
"""


# ============================================================
# 6. 1D VS 2D
# ============================================================

a = np.array([10, 20, 30])

print(a.shape)       # (3,)
print(a.reshape(-1, 1).shape)  # (3, 1)

"""
(3,)   -> 1D array
(3,1)  -> 2D column array

Many ML libraries expect feature data X to be 2D.
"""


# ============================================================
# 7. RESHAPING
# ============================================================

X_single = np.array([20, 25, 30])
X_single = X_single.reshape(-1, 1)

print(X_single) # [[20]
                #  [25]
                #  [30]]
print(X_single.shape)    # (3, 1)

"""
reshape(-1, 1)
-1 = NumPy calculates the number of rows
1  = one feature
"""


# ============================================================
# 8. SHAPE ERRORS
# ============================================================

"""
Always check:

X.shape
y.shape

For supervised ML:

X -> (number_of_samples, number_of_features)
y -> (number_of_samples,)

Example:

X.shape = (100, 5)
y.shape = (100,)

Correct:
100 samples
5 features
100 labels

If the number of samples does not match,
the model cannot correctly pair inputs with labels.
"""


# ============================================================
# 9. NORMALIZATION
# ============================================================

"""
Normalization puts values into a similar range.
Formula:
x_normalized = (x - min) / (max - min)
Example:
"""

salary = np.array([20000, 30000, 40000])
normalized_salary = (
    (salary - salary.min())
    / (salary.max() - salary.min())
)
print(normalized_salary)

"""
Output:
[0.  0.5 1. ]

This can help ML models when features have very
different numerical scales.
"""


# ============================================================
# 10. VECTORIZED CALCULATIONS
# ============================================================

# Do calculations directly on the whole array.
numbers = np.array([10, 20, 30, 40])
result = numbers * 2 + 5
print(result)

"""
No loop is required.
NumPy performs the operation on the entire array.
This is called vectorization.
Vectorization is heavily used in ML because datasets
can contain thousands or millions of values.
"""


# ============================================================
# FINAL ML CHECK
# ============================================================

"""
Before sending data to a model, you should be comfortable
with these:

✓ Convert data into NumPy arrays
✓ Separate X and y
✓ Understand samples
✓ Understand features
✓ Understand labels
✓ Understand 1D and 2D arrays
✓ Check shapes
✓ Reshape arrays
✓ Handle basic numerical transformation
✓ Understand normalization
✓ Use vectorized calculations

============================================================
NUMPY COMPLETE FOR YOUR ML ROADMAP
============================================================

You do NOT need to keep learning random advanced NumPy
features.

Next:
                PANDAS
                  ↓
        Data Cleaning & Analysis
                  ↓
        Machine Learning
============================================================
"""