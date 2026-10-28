# Boolean Maasking in NumPy


# This is very important for your AI/ML roadmap because you'll frequently need to find, filter, replace, or classify values in datasets.

"""
The main idea is:
Create a Boolean condition, then use that condition to select the values you want.

1. Boolean condition
Start with:


import numpy as np
arr = np.array([10, 20, 30, 40, 50])
Now:
arr > 25

Output:

[False False  True  True  True]

Why?

10 > 25 → False
20 > 25 → False
30 > 25 → True
40 > 25 → True
50 > 25 → True

So NumPy creates a Boolean array containing True and False.

2. Boolean masking

Now the powerful part:

arr[arr > 25]

Output:

[30 40 50]

Think of it like this:

Values:     10    20    30    40    50
Condition:  ❌    ❌    ✅    ✅    ✅
                         ↓     ↓     ↓
Result:                 30    40    50

NumPy keeps only the elements where the condition is True.

Remember:
arr[condition]

means:

Give me the elements where condition is True.

3. Even numbers
arr[arr % 2 == 0]

Output:

[10 20 30 40 50]

All these numbers are even.

The condition:

arr % 2 == 0

checks whether each number has a remainder of 0 when divided by 2.

For example:

10 % 2 → 0 → True
20 % 2 → 0 → True
30 % 2 → 0 → True
4. Another example
arr = np.array([11, 12, 15, 18, 21, 24, 30])

Find even numbers:

arr[arr % 2 == 0]

Output:

[12 18 24 30]
5. Multiple conditions

Suppose:

arr = np.array([10, 20, 30, 40, 50])

You want values:

greater than 10 AND less than 40.

Use:

arr[(arr > 10) & (arr < 40)]

Output:

[20 30]
Why &?

& means AND for NumPy Boolean arrays.

Visually:

arr > 10

10 → False
20 → True
30 → True
40 → True
50 → True

and:

arr < 40

10 → True
20 → True
30 → True
40 → False
50 → False

Both must be True:

10 → False
20 → True
30 → True
40 → False
50 → False

Result:

[20 30]
6. Important NumPy operators

For Boolean filtering, remember:

Meaning	NumPy
Equal	==
Not equal	!=
Greater than	>
Less than	<
Greater/equal	>=
Less/equal	<=
AND	&
OR	|
NOT	~

For example:

arr[(arr > 10) & (arr < 40)]
⚠️ Important

Don't use Python's normal and / or here.

❌ Don't write:

arr[(arr > 10) and (arr < 40)]

Use:

arr[(arr > 10) & (arr < 40)]

Also, put each condition inside parentheses.

7. OR condition

Suppose you want values that are:

less than 20 OR greater than 40.

arr[(arr < 20) | (arr > 40)]

Output:

[10 50]

| means OR.

8. np.where()

Now another very useful NumPy function:

np.where()

It can be used to make decisions based on a condition.

Example:

arr = np.array([10, 20, 30, 40, 50])

np.where(arr > 25, 1, 0)

Output:

[0 0 1 1 1]

The structure is:

np.where(condition, value_if_true, value_if_false)

So:

np.where(arr > 25, 1, 0)

means:

If the value is greater than 25, put 1; otherwise put 0.

Visually:

10 > 25 → False → 0
20 > 25 → False → 0
30 > 25 → True  → 1
40 > 25 → True  → 1
50 > 25 → True  → 1
9. np.where() can replace values

This is especially useful in data preprocessing.

Suppose:

arr = np.array([10, 20, 30, 40, 50])

You want every value greater than 25 to become 100.

np.where(arr > 25, 100, arr)

Output:

[10 20 100 100 100]

Meaning:

condition true  → 100
condition false → original value
10. Why this matters for Machine Learning

Imagine a dataset containing ages:

ages = np.array([15, 22, 35, 17, 45, 12, 30])

Find adults:

ages[ages >= 18]

Output:

[22 35 45 30]

Or classify them:

np.where(ages >= 18, 1, 0)

Output:

[0 1 1 0 1 0 1]

Here:

1 → Adult
0 → Not adult

The same basic ideas are useful when:

filtering datasets
detecting invalid values
removing unwanted records
creating labels
handling outliers
replacing values
preparing data before ML training
🧠 The complete mental model

There are three things you should clearly understand:

1. Create a condition
arr > 25

↓

[False False True True True]
2. Use it as a mask
arr[arr > 25]

↓

[30 40 50]
3. Use np.where() for conditional replacement/classification
np.where(arr > 25, 1, 0)

↓

[0 0 1 1 1]
🔥 Phase 7 checklist

Before moving ahead, make sure you can comfortably write and explain:

arr > 25

arr[arr > 25]

arr[arr % 2 == 0]

arr[(arr > 10) & (arr < 40)]

arr[(arr < 20) | (arr > 40)]

np.where(arr > 25, 1, 0)

np.where(arr > 25, 100, arr)

Core idea to remember:

Boolean masking selects values. np.where() makes a conditional choice or replacement.
"""