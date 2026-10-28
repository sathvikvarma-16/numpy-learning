# Statistics

"""
This phase is important because Machine Learning works heavily with statistics.
You don't need advanced statistics yet. At this stage, your goal is to understand what each measure tells you and know how to calculate it with NumPy.
"""

# Mean => Mean = average

import numpy as np
data = np.array([10, 20, 30, 40, 50])
print(np.mean(data)) # 30.0 => (10 + 20 + 30 + 40 + 50) / 5 = 30
# Mean tells you the central average of the data.


# Median -> The median is the middle value after arranging the data in order.

print(np.median(data)) # 30.0
"""
10  20  30  40  50
       ↑
     middle

So median = 30.

Why median matters
Consider:
data = np.array([10, 20, 30, 40, 1000])

Mean:
np.mean(data)
is heavily affected by 1000.

But:
np.median(data)
is still: 30
So median is more resistant to extreme values/outliers.
"""


# Variance => Variance tells you how spread out the data is from the mean.

data = np.array([10, 20, 30, 40, 50])
print(np.var(data)) # 200.0

"""
The mean is 30.

Distances from mean:
10 → -20
20 → -10
30 →   0
40 → +10
50 → +20

We square those differences:

400
100
0
100
400

Average:

1000 / 5 = 200
So variance = 200.

The important idea:
Variance measures how much the values vary around the mean.
"""



# Standard Deviation -> Standard deviation is closely related to variance.
# For ML, you'll see standard deviation frequently in standardization, normalization, feature scaling, and statistical analysis.

print(np.std(data)) # 14.142135623730951

# The relationship is : standard deviation = √variance

"""
Here:
√200 ≈ 14.14
So: np.var(data)
gives: 200

while:
np.std(data)
gives approximately:
14.14
Why do we care about standard deviation ? : It tells us roughly how far values tend to be from the mean.
"""




# minimum and maximum
data = np.array([10, 20, 30, 40, 50])
print(np.min(data))
print(np.max(data))

# or using array menthods data.min() / data.max()





# Percentile -> Percentile tells you the value below which a certain percentage of observations fall.

data = np.array([10, 20, 30, 40, 50])
print(np.percentile(data, 50)) # 30.0 => The 50th percentile is essentially the median.
np.percentile(data, 25) # 20.0 => the 25th percentile is the value below which 25% of the data falls.
np.percentile(data, 75) # 40.0 => the 75th percentile is the value below which 75% of the data falls.

"""
A useful way to visualize it:
10 ---- 20 ---- 30 ---- 40 ---- 50
       ↑        ↑        ↑
      25%      50%      75%

Percentiles become particularly useful later when you learn quartiles, IQR, and outlier detection.
"""



# Your complete NumPy statistics toolkit

# You should be comfortable with:
np.mean(data)
np.median(data)
np.var(data)
np.std(data)

np.min(data)
np.max(data)

np.percentile(data, 25)
np.percentile(data, 50)
np.percentile(data, 75)



# Statistics with axis -> This connects directly to Phase 5: Axis.

# Suppose:
data = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50]
])

# Mean of each column:
np.mean(data, axis=0) # [20. 30. 40.]

# Mean of each row:
np.mean(data, axis=1) # [20. 30. 40.]

"""
Again, the values happen to be the same here, but the meaning is different.
This is why your Phase 5 axis knowledge matters.
"""


# Why statistics matter for your ML path

"""
Imagine your dataset has:

Age
Salary
Experience

You might use:
Mean → understand average values
Median → understand the center when outliers exist
Variance → measure spread
Standard deviation → measure typical spread
Min/Max → understand range
Percentiles → understand distribution and detect unusual values

Later, these concepts will appear in:

NumPy
   ↓
Pandas
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Feature Scaling
   ↓
Machine Learning

🔥 What you need to master now
Don't go deep into statistics yet.
For NumPy, make sure you can explain:
Mean = average
Median = middle value
Variance = measure of spread around the mean
Standard deviation = square root of variance, describing spread in the original units
Percentile = value marking a position in the data distribution
And be able to write:

np.mean()
np.median()
np.var()
np.std()
np.min()
np.max()
np.percentile()

That's enough for Phase 8. You do not need probability distributions, hypothesis testing, regression statistics, or advanced statistical theory yet.
"""