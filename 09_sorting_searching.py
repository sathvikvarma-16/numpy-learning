# PHASE 9: SORTING AND SEARCHING IN NUMPY

import numpy as np

"""
PHASE 9 GOAL
------------

In this phase, we will learn how to:

1. Sort NumPy arrays
2. Understand np.sort()
3. Understand np.argsort()
4. Find the index of the maximum value
5. Find the index of the minimum value
6. Find top values
7. Rank data
8. Sort 2D arrays
9. Sort along rows and columns using axis
10. Use argsort() for ranking and top-k selection
11. Use these concepts in Machine Learning

The main functions are:

```
np.sort()
np.argsort()
np.argmax()
np.argmin()
```

The most important difference to remember is:

```
np.sort()   -> gives SORTED VALUES

np.argsort() -> gives INDICES that would sort the array

np.argmax() -> gives INDEX of maximum value

np.argmin() -> gives INDEX of minimum value
```

"""

# ============================================================
# 1. BASIC ARRAY
# ============================================================

arr = np.array([30, 10, 20])
print(arr)

"""
Output:
[30 10 20]
The array is currently NOT sorted.
"""

# ============================================================
# 2. np.sort()
# ============================================================

"""
np.sort() sorts the values of an array.

Syntax:

```
np.sort(array)
```

It returns a NEW sorted array.
"""

arr = np.array([30, 10, 20])
sorted_arr = np.sort(arr)
print(sorted_arr)

"""
Output:
[10 20 30]
The original array is still unchanged.
"""

print(arr)

"""
Output:
[30 10 20]

IMPORTANT:
np.sort(arr) does NOT normally modify the original array.
It returns a sorted copy.
"""

# ============================================================
# 3. SORTING IN DESCENDING ORDER
# ============================================================

"""
np.sort() sorts in ascending order by default.
Ascending:
```
10 20 30 40 50
```

Descending:
```
50 40 30 20 10
```

NumPy does not directly provide a descending option for np.sort()
like some Python sorting functions.
A common method is:

```
np.sort(arr)[::-1]
```

[::-1] reverses the array.
"""

arr = np.array([30, 10, 50, 20, 40])
descending = np.sort(arr)[::-1]
print(descending)

"""
Output:
[50 40 30 20 10]
"""

# ============================================================
# 4. np.argsort()
# ============================================================

"""
THIS IS VERY IMPORTANT.
np.argsort() does NOT return the sorted values.
Instead, it returns the INDICES that would sort the array.

Example:

```
arr = [30, 10, 20]
```

Indices:

```
index:    0   1   2
value:   30  10  20
```

To sort the values:

```
10 -> index 1
20 -> index 2
30 -> index 0
```

Therefore:

```
np.argsort(arr)
```

returns:

```
[1 2 0]
```

"""

arr = np.array([30, 10, 20])
indices = np.argsort(arr)
print(indices)

"""
Output:
[1 2 0]
"""

# ============================================================
# 5. UNDERSTANDING argsort() VISUALLY
# ============================================================

arr = np.array([30, 10, 20])
print("Values :", arr)
print("Indices:", np.arange(len(arr)))

"""
We have:

Index:
0   1   2
Value:
30  10  20

Sorted values:
```
10  20  30
```

Their original indices are:
```
1   2   0
```

Therefore:

```
np.argsort(arr)
```

gives:

```
[1 2 0]
```

"""

print(np.argsort(arr))
# Use those indices to actually get sorted values:
print(arr[np.argsort(arr)])

"""
Output:
[10 20 30]
This is an extremely important relationship:

```
arr[np.argsort(arr)]
```

gives the sorted array.
"""


# ============================================================
# 6. SORT VS ARGSORT
# ============================================================

arr = np.array([50, 10, 40, 20, 30])
print(np.sort(arr))
print(np.argsort(arr))

"""
np.sort(arr):

```
[10 20 30 40 50]
```

This gives the VALUES.
np.argsort(arr):

```
[1 3 4 2 0]
```

This gives the INDICES.
Let's verify:

Index:
0   1   2   3   4

Value:
50  10  40  20  30

Sorted:

10 -> index 1
20 -> index 3
30 -> index 4
40 -> index 2
50 -> index 0

Therefore:

```
[1 3 4 2 0]
```

"""

# ============================================================
# 7. np.argmax()
# ============================================================

"""
np.argmax() returns the INDEX of the maximum value.

Syntax:
```
np.argmax(array)
```
"""

arr = np.array([10, 50, 20, 40, 30])
index = np.argmax(arr)
print(index) # 1

"""
Why?

Array:
```
Index:   0   1   2   3   4
Value:  10  50  20  40  30
```

Maximum value = 50
50 is at index 1.

Therefore:
```
np.argmax(arr) -> 1
```

"""


# ============================================================
# 8. GET THE MAXIMUM VALUE USING argmax()
# ============================================================

arr = np.array([10, 50, 20, 40, 30])
index = np.argmax(arr)
maximum = arr[index]
print(maximum)

"""
Output:
50
Remember:
```
np.argmax(arr)
    ->
index of maximum value
```

It does NOT directly return the maximum value.
For the maximum value itself:
```
np.max(arr)
```

"""


# ============================================================
# 9. np.max() VS np.argmax()
# ============================================================

arr = np.array([10, 50, 20, 40, 30])
print(np.max(arr)) # 50
print(np.argmax(arr)) # 1

"""
np.max():

```
Gives the maximum VALUE.
```

np.argmax():

```
Gives the INDEX where the maximum value is located.
```

Remember:

```
max     -> WHAT is the maximum?
argmax  -> WHERE is the maximum?
```

"""

# ============================================================
# 10. np.argmin()
# ============================================================

"""
np.argmin() returns the INDEX of the minimum value.
"""

arr = np.array([10, 50, 20, 40, 30])
index = np.argmin(arr)
print(index) # 0

"""
The minimum value is 10.
10 is at index 0.
"""

# ============================================================
# 11. np.min() VS np.argmin()
# ============================================================

arr = np.array([10, 50, 20, 40, 30])
print(np.min(arr))
print(np.argmin(arr))

"""
np.min():

```
Returns minimum VALUE.
```
np.argmin():

```
Returns INDEX of minimum value.
```

Remember:

```
min     -> WHAT is minimum?
argmin  -> WHERE is minimum? => gives the index
```

"""
# ============================================================
# 12. SIMPLE MEMORY TRICK
# ============================================================


"""
The word "arg" can be thought of as "argument/index position".
So:
```
np.argmax()
    -> index of maximum

np.argmin()
    -> index of minimum
```

Compare:

```
np.max()       -> maximum VALUE
np.argmax()    -> maximum INDEX

np.min()       -> minimum VALUE
np.argmin()    -> minimum INDEX
```


"""
# ============================================================
# 13. FINDING THE TOP VALUE
# ============================================================

scores = np.array([72, 95, 81, 88, 67])
highest_index = np.argmax(scores)
highest_score = scores[highest_index]
print(highest_score) # 95

"""
Steps:

1. np.argmax(scores)
   finds the index of the highest score.

2. scores[index]
   gets the actual score.
   """


# ============================================================
# 14. FIND THE STUDENT WITH THE HIGHEST SCORE
# ============================================================

scores = np.array([72, 95, 81, 88, 67])
names = np.array([
"A",
"B",
"C",
"D",
"E"
])
highest_index = np.argmax(scores)
print(names[highest_index]) # B
print(scores[highest_index]) # 95

"""

This is a very useful pattern.
If two arrays have matching positions:
```
names
scores
```

we can use the index returned by argmax() to identify
which name belongs to the highest score.
"""


# ============================================================
# 15. FIND LOWEST SCORE
# ============================================================


lowest_index = np.argmin(scores)
print(names[lowest_index]) # E
print(scores[lowest_index]) # 67


# ============================================================
# 16. RANKING USING argsort()
# ============================================================


scores = np.array([72, 95, 81, 88, 67])
order = np.argsort(scores)
print(order) # [4 0 2 3 1]

"""
The result contains indices in ascending order.

Scores:
Index:    0   1   2   3   4
Score:   72  95  81  88  67

Sorted:

67 -> index 4
72 -> index 0
81 -> index 2
88 -> index 3
95 -> index 1

Therefore:

"""

print(scores[order]) # [67 72 81 88 95]




# ============================================================
# 17. RANKING NAMES WITH SCORES
# ============================================================


scores = np.array([72, 95, 81, 88, 67])
names = np.array([
"A",
"B",
"C",
"D",
"E"
])
order = np.argsort(scores)
print(names[order]) # ['E' 'A' 'C' 'D' 'B']

"""
This means:
1. E -> 67
2. A -> 72
3. C -> 81
4. D -> 88
5. B -> 95
"""



# ============================================================
# 18. DESCENDING RANKING
# ============================================================


order = np.argsort(scores)[::-1]
print(order)
print(names[order]) # ['B' 'D' 'C' 'A' 'E']
print(scores[order]) # [95 88 81 72 67]


"""
This is a very useful pattern:

```
np.argsort(scores)[::-1]
```

means:
```
Find sorting indices
+
Reverse them
```

Therefore we get descending order.
"""



# ============================================================
# 19. TOP 3 VALUES
# ============================================================

scores = np.array([72, 95, 81, 88, 67])
order = np.argsort(scores)[::-1]
top_3_indices = order[:3]
print(top_3_indices)
print(scores[top_3_indices])

"""
Output:
[95 88 81]

The process:
1. argsort()
2. reverse using [::-1]
3. take first 3 using [:3]
   """

# ============================================================
# 20. TOP 3 STUDENTS
# ============================================================

scores = np.array([72, 95, 81, 88, 67])
names = np.array([
"A",
"B",
"C",
"D",
"E"
])
order = np.argsort(scores)[::-1]
top_3 = order[:3]
print(names[top_3])
print(scores[top_3])

"""
Output:
['B' 'D' 'C']
[95 88 81]


Therefore:

1. B -> 95
2. D -> 88
3. C -> 81
   """



# ============================================================
# 21. FINDING TOP-K VALUES
# ============================================================


"""
K means how many values you want.
For example:

```
Top 1
Top 3
Top 5
Top 10
```

General pattern:

```
order = np.argsort(arr)[::-1]
top_k = order[:k]
```

Then:
```
arr[top_k]
```

gives the top k values.
"""

arr = np.array([15, 92, 31, 76, 54, 88])
k = 3
order = np.argsort(arr)[::-1]
top_k_indices = order[:k] # [1 5 3] -> gives indices of top 3 values because we reversed the order
print(arr[top_k_indices])


# ============================================================
# 22. 2D ARRAYS AND np.sort()
# ============================================================

matrix = np.array([
[30, 10, 20],
[60, 40, 50]
])
print(np.sort(matrix))

"""
Output:

[[10 20 30]
[40 50 60]]

By default, NumPy sorts each row.
So:
```
[30 10 20]
    ->
[10 20 30]
```

and:
```
[60 40 50]
    ->
[40 50 60]
```
"""



# ============================================================
# 23. np.sort() WITH axis=0
# ============================================================


matrix = np.array([
[30, 10, 20],
[60, 40, 50]
])
print(np.sort(matrix, axis=0))

"""
Output:

[[30 10 20]
[60 40 50]]
Here each column is sorted.

Column 1:
```
30
60
```
already sorted.

Column 2:
```
10
40
```
already sorted.
Column 3:
```
20
50
```
already sorted.
"""



# ============================================================
# 24. ANOTHER axis=0 EXAMPLE
# ============================================================

matrix = np.array([
[60, 20, 90],
[10, 80, 30],
[50, 40, 70]
])

print(np.sort(matrix, axis=0))
"""
Original:
60 20 90
10 80 30
50 40 70

Sort DOWN each column:
Column 1:
60
10
50
becomes:
10
50
60

Column 2:
20
80
40
becomes :
20
40
80

Column 3:
90
30
70
becomes:
30
70
90

Result:
10 20 30
50 40 70
60 80 90
"""



# ============================================================
# 25. axis=1
# ============================================================

matrix = np.array([
[60, 20, 90],
[10, 80, 30],
[50, 40, 70]
])

print(np.sort(matrix, axis=1))

"""
axis=1 means sort ACROSS each row.

Original:

60 20 90
10 80 30
50 40 70

Result:

20 60 90
10 30 80
40 50 70
"""


# ============================================================
# 26. np.argmax() WITH 2D ARRAY
# ============================================================

matrix = np.array([
[10, 50, 30],
[80, 20, 60]
])
print(np.argmax(matrix))

"""
Output:
3

Without specifying axis, np.argmax() treats the array
as if it were flattened.

Flattened array:
[10, 50, 30, 80, 20, 60]

Indices:
0   1   2   3   4   5
Maximum = 80
Index = 3
"""


# ============================================================
# 27. np.argmax() axis=0
# ============================================================

matrix = np.array([
[10, 50, 30],
[80, 20, 60]
])
print(np.argmax(matrix, axis=0))

"""
Output:

[1 0 1]
We are finding the maximum INDEX in each column.
Column 1:
10
80
Maximum = 80
Row index = 1
Column 2:
50
20
Maximum = 50
Row index = 0
Column 3:
30
60
Maximum = 60
Row index = 1
Therefore:
[1 0 1]
"""



# ============================================================
# 28. np.argmax() axis=1
# ============================================================


print(np.argmax(matrix, axis=1))
"""
Output:
[1 0]
We are finding the maximum INDEX in each row.
Row 1:
10 50 30
Maximum = 50
Column index = 1
Row 2:
80 20 60
Maximum = 80
Column index = 0
Therefore:
[1 0]
"""


# ============================================================
# 29. np.argmin() WITH axis
# ============================================================

matrix = np.array([
[10, 50, 30],
[80, 20, 60]
])
print(np.argmin(matrix, axis=0))
"""
Find minimum index in each column.
Column 1:
10
80
Minimum = 10
Index = 0
Column 2:
50
20
Minimum = 20
Index = 1
Column 3:
30
60
Minimum = 30
Index = 0

Result:
[0 1 0]
"""



# ============================================================
# 30. IMPORTANT: argmax() RETURNS AN INDEX, NOT A VALUE
# ============================================================


arr = np.array([100, 20, 300, 40])
index = np.argmax(arr)
print(index)
"""
Output:
2

The maximum VALUE is:
300
The maximum VALUE is at:
index 2
Therefore:
```
np.argmax(arr) -> 2
```

NOT:
```
300
```

"""
print(arr[index])

"""
Output:
300
"""



# ============================================================
# 31. WORKING WITH PREDICTIONS
# ============================================================

"""
This is directly relevant to Machine Learning.
Suppose a model predicts probabilities for three classes:
```
Cat
Dog
Horse
```

For one image:

```
[0.10, 0.75, 0.15]
```

The largest probability is:

```
0.75
```

Its index is:

```
1
```

Therefore the predicted class is:

```
Dog
```

"""
probabilities = np.array([0.10, 0.75, 0.15])
predicted_class = np.argmax(probabilities)
print(predicted_class)
"""
Output:
1
If:
```
0 -> Cat
1 -> Dog
2 -> Horse
```
then:
```
prediction = Dog
```
"""



# ============================================================
# 32. ML EXAMPLE WITH CLASS NAMES
# ============================================================

classes = np.array([
"Cat",
"Dog",
"Horse"
])
probabilities = np.array([
0.10,
0.75,
0.15
])
index = np.argmax(probabilities)
prediction = classes[index]
print(prediction) # Dog

"""
This is one of the most important practical uses
of np.argmax() in Machine Learning.
A classification model can output probabilities.
np.argmax() tells us which class has the highest probability.
"""


# ============================================================
# 33. TOP 2 PREDICTIONS
# ============================================================


classes = np.array([
"Cat",
"Dog",
"Horse"
])
probabilities = np.array([
0.10,
0.75,
0.15
])

order = np.argsort(probabilities)[::-1]
top_2 = order[:2]
print(classes[top_2])
print(probabilities[top_2])


"""
Output:
['Dog' 'Horse']
[0.75 0.15]
So the model's:

Top 1 prediction:
Dog -> 75%

Top 2 prediction:
Horse -> 15%

argsort() is useful when we need more than just the
single highest prediction.
"""


# ============================================================
# 34. SORTING MULTIPLE RELATED ARRAYS
# ============================================================


products = np.array([
"Laptop",
"Phone",
"Tablet",
"Watch"
])

prices = np.array([
70000,
40000,
30000,
15000
])

order = np.argsort(prices)[::-1]
print(products[order])
print(prices[order])

"""
Output:
Products sorted by price from highest to lowest.
Laptop -> 70000
Phone  -> 40000
Tablet -> 30000
Watch  -> 15000
This works because all arrays use matching indices.
"""



# ============================================================
# 35. IMPORTANT PATTERN: SORT ONE ARRAY, REORDER ANOTHER
# ============================================================


"""
Suppose:
names contain labels
scores contain values
We don't want to sort names alphabetically.
We want to sort names according to scores.
Therefore:
1. Find sorting indices from scores.
2. Use those indices on names.
   """


names = np.array(["A", "B", "C", "D"])
scores = np.array([70, 95, 80, 85])
order = np.argsort(scores)[::-1]
print(names[order])
print(scores[order])

"""
Output:

B 95
D 85
C 80
A 70
"""




# ============================================================
# 36. WHAT HAPPENS WITH DUPLICATE VALUES?
# ============================================================

arr = np.array([30, 10, 30, 20, 10])
print(np.sort(arr))

"""
Output:
[10 10 20 30 30]

np.sort() correctly handles duplicate values.
"""



# ============================================================
# 37. np.argmax() WITH DUPLICATES
# ============================================================


arr = np.array([50, 20, 50, 10])
print(np.argmax(arr)) # 0 -> return the index of the first occurrence of the maximum value
"""
There are two maximum values:
50 at index 0
50 at index 2
np.argmax() returns the FIRST occurrence.
"""



# ============================================================
# 38. np.argmin() WITH DUPLICATES
# ============================================================


arr = np.array([10, 20, 10, 30])
print(np.argmin(arr))
"""
Minimum value = 10
It appears at:
index 0
index 2
np.argmin() returns the first occurrence:
0
"""



# ============================================================
# 39. SORTING STRINGS
# ============================================================


names = np.array([
"Charlie",
"Alice",
"Bob"
])
print(np.sort(names)) # ['Alice' 'Bob' 'Charlie']
"""
Output:
['Alice' 'Bob' 'Charlie']
NumPy can also sort strings.
"""


# ============================================================
# 40. PRACTICAL DATA ANALYSIS EXAMPLE
# ============================================================

students = np.array([
"S1",
"S2",
"S3",
"S4",
"S5"
])
marks = np.array([
78,
92,
85,
96,
70
])


"""
Question 1:
Who scored the highest?
"""
highest = np.argmax(marks)
print(students[highest]) # s4
print(marks[highest]) # 96


"""
Answer:
S4
96
"""

"""
Question 2:
Who scored the lowest?
"""
lowest = np.argmin(marks)
print(students[lowest]) # s5
print(marks[lowest]) # 70

"""
Answer:

S5
70
"""

"""
Question 3:
Rank all students from highest to lowest.
"""

ranking = np.argsort(marks)[::-1]
print(students[ranking])
print(marks[ranking])

"""
Ranking:

S4 -> 96
S2 -> 92
S3 -> 85
S1 -> 78
S5 -> 70
"""

# ============================================================
# 41. TOP 3 STUDENTS
# ============================================================

top_3 = ranking[:3]
print(students[top_3])
print(marks[top_3])

"""
Output:
S4
S2
S3

Marks:
96
92
85
"""


# ============================================================
# 42. BOTTOM 2 STUDENTS
# ============================================================


bottom_2 = ranking[-2:]
print(students[bottom_2])
print(marks[bottom_2])

"""
Be careful:
ranking is descending:
S4 S2 S3 S1 S5
[-2:] gives:
S1 S5
If you want the bottom students from lowest to highest,
you can use:
ascending = np.argsort(marks)
bottom_2 = ascending[:2]
"""



# ============================================================
# 43. BOTTOM 2 FROM LOWEST TO HIGHEST
# ============================================================

ascending = np.argsort(marks)
bottom_2 = ascending[:2]
print(students[bottom_2])
print(marks[bottom_2])


"""
Output:
S5 -> 70
S1 -> 78
"""


# ============================================================
# 44. SORTING WITH axis AND 2D DATA
# ============================================================

data = np.array([
[30, 90, 70],
[80, 40, 60],
[50, 20, 100]
])
# maths | science | english

"""
Imagine each row represents a student
and each column represents a subject.

Columns:
```
Math
Science
English
```

"""
print(np.sort(data, axis=1))
"""
Sort each student's marks across subjects.
Each row becomes sorted independently.
"""


print(np.sort(data, axis=0))
"""
Sort each subject DOWN the r
ows.
Each column becomes sorted independently.
"""



# ============================================================
# 45. argsort() WITH axis
# ============================================================


data = np.array([
[30, 90, 70],
[80, 40, 60],
[50, 20, 100]
])
print(np.argsort(data, axis=1))


"""
This gives the indices that would sort EACH ROW.
It does not give sorted values.
For example:
Row 1:
[30, 90, 70]
Sorted:
[30, 70, 90]
Indices:
[0, 2, 1]
"""



# ============================================================
# 46. SORTING A 2D ARRAY USING argsort()
# ============================================================


order = np.argsort(data, axis=1)
print(order)
"""
The result contains indices for each row.
You can use these indices to reorder the values.
The important idea is:

```
argsort()
    ->
sorting positions
```
while:
```
sort()
    ->
sorted values
```

"""



# ============================================================
# 47. KEY DIFFERENCE TABLE
# ============================================================

"""
FUNCTION             WHAT IT RETURNS
np.sort(arr)
-> Sorted VALUES

np.argsort(arr)
-> INDICES that would sort the array

np.max(arr)
-> Maximum VALUE

np.argmax(arr)
-> INDEX of maximum VALUE

np.min(arr)
-> Minimum VALUE

np.argmin(arr)
-> INDEX of minimum VALUE

Example:

arr = [30, 10, 20]

np.sort(arr)
-> [10 20 30]

np.argsort(arr)
-> [1 2 0]

np.max(arr)
-> 30

np.argmax(arr)
-> 0

np.min(arr)
-> 10

np.argmin(arr)
-> 1
"""



# ============================================================
# 48. THE MOST IMPORTANT PATTERNS TO REMEMBER
# ============================================================



"""
PATTERN 1:
Sort values
```
np.sort(arr)
```



PATTERN 2:
Get sorting indices

```
np.argsort(arr)
```


PATTERN 3:
Actually sort using argsort
```
arr[np.argsort(arr)]
```



PATTERN 4:
Highest value
```
np.max(arr)
```


PATTERN 5:
Index of highest value
```
np.argmax(arr)
```



PATTERN 6:
Lowest value
```
np.min(arr)
```



PATTERN 7:
Index of lowest value
```
np.argmin(arr)
```


PATTERN 8:
Descending order
```
np.sort(arr)[::-1]
```


PATTERN 9:
Descending sorting indices
```
np.argsort(arr)[::-1]
```


PATTERN 10:
Top K values
```
order = np.argsort(arr)[::-1]
top_k = order[:k]
arr[top_k]
```
"""


# ============================================================
# 49. MACHINE LEARNING CONNECTION
# ============================================================
"""


Why are these functions important for ML?
Suppose a classification model gives:
```
[0.05, 0.80, 0.10, 0.05]
```


These could represent:
```
Cat      -> 0.05
Dog      -> 0.80
Horse    -> 0.10
Bird     -> 0.05
```

np.argmax() gives:
```
1
```
Therefore:
```
Dog
```
is the predicted class.
If we need the top 3 predictions:
```
np.argsort(probabilities)[::-1][:3]
```


gives the indices of the three highest probabilities.
This is useful in:
```
Classification
Recommendation systems
Ranking
Search
Data analysis
Model predictions
```
"""


# ============================================================
# 50. FINAL PRACTICE
# ============================================================

"""
DO THESE YOURSELF WITHOUT LOOKING AT THE ANSWERS.

Question 1:
"""
arr = np.array([40, 10, 70, 20, 90, 30])
# Find sorted values.
# Write your code below.
"""


Question 2:
Find the indices that would sort arr.
"""

"""
Question 3:
Find the maximum value.
"""


"""
Question 4:
Find the index of the maximum value.
"""


"""
Question 5:
Find the minimum value.
"""


"""
Question 6:
Find the index of the minimum value.
"""


"""
Question 7:
Find the top 3 values.
"""


"""
Question 8:
Find the indices of the top 3 values.
"""


"""
Question 9:
Create this array:
[95, 70, 85, 60, 90]
Find the students/positions from highest score to lowest.
"""

"""
Question 10:
Given:
probabilities = np.array([0.05, 0.70, 0.15, 0.10])

Find the index of the predicted class.
"""


# ============================================================
# PHASE 9 SUMMARY
# ============================================================

"""
FINAL SUMMARY
-------------

1. np.sort()
   Sorts VALUES.

2. np.argsort()
   Returns INDICES that would sort the array.

3. np.argmax()
   Returns INDEX of the maximum value.

4. np.argmin()
   Returns INDEX of the minimum value.

5. np.max()
   Returns maximum VALUE.

6. np.min()
   Returns minimum VALUE.

7. [::-1]
   Reverses an array.

8. Top K pattern:

   ```
   order = np.argsort(arr)[::-1]
   top_k = order[:k]
   arr[top_k]
   ```

9. ML prediction pattern:

   ```
   prediction = np.argmax(probabilities)
   ```

10. Ranking related data:

    order = np.argsort(scores)[::-1]
    names[order]
    scores[order]

MOST IMPORTANT MEMORY RULE:

```
SORT     -> VALUES
ARGSORT  -> INDICES

MAX      -> VALUE
ARGMAX   -> INDEX

MIN      -> VALUE
ARGMIN   -> INDEX
```

If you understand these four pairs clearly,
you have the core of NumPy sorting and searching.
"""
