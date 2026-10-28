# Axis Concept

import numpy as np
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

"""
Think of it as a table:

        Column
          ↓
       1   2   3
     ┌─────────────
Row 1│  1   2   3
Row 2│  4   5   6        """



# Shape:
arr.shape   # Output: (2, 3) => That means: 2 rows × 3 columns

# 1) No axis
np.sum(arr) # NumPy adds everything: 1 + 2 + 3 + 4 + 5 + 6 => Output: 21

"""
Think:

[1 2 3]
[4 5 6]
 ↓ ↓ ↓
Everything

2. axis=0 → Down the rows
This is the one students commonly confuse.
np.sum(arr, axis=0)
Look vertically:

  1   2   3
  +   +   +
  4   5   6
  ↓   ↓   ↓
  5   7   9

Calculations:
Column 1: 1 + 4 = 5
Column 2: 2 + 5 = 7
Column 3: 3 + 6 = 9

Output:

[5 7 9]
Remember:

axis=0 means collapse the rows, keeping the columns.

So:

2 × 3
 ↓ axis=0
   3

Result shape: (3,)
3. axis=1 → Across the columns
np.sum(arr, axis=1)

Now look horizontally:

1 + 2 + 3 → 6

4 + 5 + 6 → 15

Output: [6 15]

Visually:
[1  2  3] → 6
[4  5  6] → 15

Remember:
axis=1 means collapse the columns, keeping the rows.

So:
2 × 3
 ↓ axis=1
   2

Result shape:
(2,)

🧠 The easiest way to remember
Don't memorize:

axis 0 = columns
axis 1 = rows

That wording can cause confusion.

Instead remember:
axis=0 → ↓ DOWN
1   2   3
↓   ↓   ↓
4   5   6

You are combining values downward.

Result: [5 7 9]

axis=1 → → ACROSS
1 → 2 → 3
4 → 5 → 6

You are combining values across.

Result: [6 15]

4. More examples
Mean->np.mean(arr, axis=0)

Column means:

(1 + 4) / 2 = 2.5
(2 + 5) / 2 = 3.5
(3 + 6) / 2 = 4.5

Output: [2.5 3.5 4.5]
np.mean(arr, axis=1)

Row means:
(1 + 2 + 3) / 3 = 2
(4 + 5 + 6) / 3 = 5

Output: [2. 5.]

5. Maximum
np.max(arr, axis=0)

Column-wise maximum: [4 5 6]
np.max(arr, axis=1)

Row-wise maximum: [3 6]

"""



"""
6. Why this matters for ML

Suppose you have:

       Feature 1  Feature 2  Feature 3
Sample 1    10        20        30
Sample 2    20        30        40
Sample 3    30        40        50

NumPy array:

X = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50]
])

If you want the average of each feature:

np.mean(X, axis=0)
You get: [20. 30. 40.]

That's extremely relevant to machine-learning preprocessing, where operations are often performed feature-wise.
If you want the average of each sample:

np.mean(X, axis=1)
You get: [20. 30. 40.]

Same numbers here by coincidence, but the meaning is different.

🔥 Your Phase 5 rule

For a 2D array:

axis=0 → ↓ DOWN → operate vertically → keep columns
axis=1 → → ACROSS → operate horizontally → keep rows
And always ask yourself:

"What direction am I collapsing?"
If you can visualize that, you've actually understood axis, rather than memorizing it.'

"""


# Practice this before moving on

arr = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# Predict these without running Python:

np.sum(arr, axis=0) # [ 120 150 180] → 10+40+70, 20+50+80, 30+60+90
np.sum(arr, axis=1) #  [ 60 150 240] → 10+20+30, 40+50+60, 70+80+90

np.mean(arr, axis=0) # [ 40. 50. 60.]
np.mean(arr, axis=1) # [ 20. 50. 80.]

np.max(arr, axis=0) # [ 70 80 90]
np.max(arr, axis=1) # [ 30 60 90]

# If you can get all six correctly, your 2D axis fundamentals are solid.
