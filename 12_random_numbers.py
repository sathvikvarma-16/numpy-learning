# ============================================================

# PHASE 12: RANDOM NUMBERS IN NUMPY

# ============================================================

import numpy as np

"""
PHASE 12 GOAL
-------------

In this phase, we will learn:

```
1. What random numbers are
2. np.random.rand()
3. np.random.randn()
4. np.random.randint()
5. np.random.choice()
6. Random numbers with a seed
7. Reproducibility
8. The modern NumPy Generator
9. np.random.default_rng()
10. rng.integers()
11. rng.random()
12. rng.normal()
13. rng.choice()
14. Random arrays
15. Random sampling
16. Shuffling
17. Why randomness is important in ML
```

IMPORTANT:
There are TWO styles you will see.
Older/common style:

```
np.random.rand()
np.random.randn()
np.random.randint()
np.random.choice()

Modern recommended style:
rng = np.random.default_rng(42)

rng.random()
rng.normal()
rng.integers()
rng.choice()


For new code, prefer the Generator style.
"""


# ============================================================
# 1. WHAT ARE RANDOM NUMBERS?
# ============================================================

"""
A random number is a number generated in a way that
is not directly predictable from the previous number.
For example:

```
0.23
0.81
0.47
0.05
```

These numbers can be useful for:
```
simulations
experiments
sampling
testing
ML initialization
creating synthetic data
random selection
```


Computers actually generate PSEUDO-RANDOM numbers.
That means:
They look random,
but they are generated using an algorithm.
"""

# ============================================================
# 2. np.random.rand()
# ============================================================

"""
np.random.rand() generates random numbers from a
uniform distribution between:
0 (inclusive)
and
1 (exclusive)

Example:
"""
number = np.random.rand()
print(number)
"""
Possible output:
0.37454011

Every time you run the program, you will normally
get a different value.
Example:
Run 1 -> 0.37
Run 2 -> 0.81
Run 3 -> 0.12

The exact values are not important.
"""


# ============================================================
# 3. np.random.rand() WITH MULTIPLE VALUES
# ============================================================


numbers = np.random.rand(5)
print(numbers)
"""
Possible output:
[0.23 0.81 0.45 0.12 0.67]
5 random values were generated.
Shape: (5,)
"""


# ============================================================
# 4. np.random.rand() WITH 2D SHAPE
# ============================================================


matrix = np.random.rand(2, 3)
print(matrix)

"""
Possible output:
[[0.15 0.72 0.43]
[0.91 0.28 0.61]]

Shape: (2, 3)

This means:
2 rows
3 columns
Every value is between 0 and 1.
"""


# ============================================================
# 5. np.random.rand() AND UNIFORM DISTRIBUTION
# ============================================================

"""
Uniform distribution means that values are generated
across the specified range without favoring one region
of the range in the way a normal distribution does.

np.random.rand():
Range = [0, 1)

Meaning:
0 can occur
1 is not included
"""

# ============================================================
# 6. np.random.randn()
# ============================================================

"""
np.random.randn() generates random numbers from a
STANDARD NORMAL DISTRIBUTION.
Also called: Gaussian distribution
The standard normal distribution has:

Mean = 0
Standard deviation = 1

Values can be:

negative
zero
positive
"""

number = np.random.randn()
print(number)

"""
Possible output:
-0.42
1.37
0.08
The value is NOT restricted to 0 and 1.
"""


# ============================================================
# 7. np.random.randn() MULTIPLE VALUES
# ============================================================

numbers = np.random.randn(5)
print(numbers)  # [-0.23  1.21 -0.56  0.17  0.89]

"""
5 values were generated.
They come from a standard normal distribution.
"""


# ============================================================
# 8. np.random.randn() 2D ARRAY
# ============================================================

matrix = np.random.randn(2, 3)
print(matrix)

"""
Possible output:

[
[ 0.25, -1.12,  0.43],
[ 1.42,  0.05, -0.78]
]

Shape: (2, 3) -> The values can be negative or positive.
"""


# ============================================================
# 9. rand() VS randn()
# ============================================================

"""
VERY IMPORTANT:
np.random.rand()
Uniform distribution
Values between 0 and 1.


np.random.randn()
Standard normal distribution
Mean = 0
Standard deviation = 1
Values can be negative or positive.

MEMORY:
rand  -> uniform [0, 1)
randn -> normal distribution
"""


# ============================================================
# 10. np.random.randint()
# ============================================================

"""
np.random.randint() generates random INTEGER values.
Syntax:
np.random.randint(low, high)

IMPORTANT:
low is included
high is EXCLUDED
"""

number = np.random.randint(1, 10)
print(number)

"""
Possible values:
1
2
3
...
9
10 is NOT included.
"""



# ============================================================
# 11. randint() MULTIPLE VALUES
# ============================================================


numbers = np.random.randint(1, 10, size=5)
print(numbers)

"""
Possible output:
[3 8 1 6 4]
5 random integers were generated.

Range:
1 to 9
"""


# ============================================================
# 12. randint() WITH A 2D SHAPE
# ============================================================


matrix = np.random.randint(
1,
10,
size=(2, 3)
)
print(matrix)

"""
Possible output:

[[2 7 4]
[9 1 6]]

Shape: (2, 3)
All values are integers from:
1 to 9
"""


# ============================================================
# 13. randint() WITH 0
# ============================================================

numbers = np.random.randint(0, 5, size=10)
print(numbers)

"""
Possible values:
0
1
2
3
4

5 is excluded.
"""


# ============================================================
# 14. np.random.choice()
# ============================================================

"""
np.random.choice() randomly selects values
from an array.
Example:
"""

arr = np.array([10, 20, 30, 40, 50])
selected = np.random.choice(arr)
print(selected)
"""
Possible output: 20 or: 40 or: 10
One element was randomly selected.
"""


# ============================================================
# 15. choice() MULTIPLE VALUES
# ============================================================

arr = np.array([10, 20, 30, 40, 50])
selected = np.random.choice(arr, size=3)
print(selected)

"""
Possible output : [20 50 10]
3 values were randomly selected.
"""



# ============================================================
# 16. choice() WITH REPLACEMENT
# ============================================================

arr = np.array([10, 20, 30])
selected = np.random.choice(
arr,
size=5,
replace=True
)
print(selected)

"""
Possible output : [20 20 10 30 20]

replace=True means:
After selecting a value,
it can be selected again.
So duplicates are allowed.
"""


# ============================================================
# 17. choice() WITHOUT REPLACEMENT
# ============================================================

arr = np.array([10, 20, 30, 40, 50])
selected = np.random.choice(
arr,
size=3,
replace=False
)
print(selected)

"""
Possible output:
[40 10 30]
replace=False means:

Once an element is selected,
it cannot be selected again.

Therefore : No duplicates.
"""


# ============================================================
# 18. REPLACEMENT CONCEPT
# ============================================================


"""
WITH replacement : [1, 2, 3]

Choose 5 values : [2, 2, 1, 3, 2]

Duplicates are allowed.
WITHOUT replacement:
[1, 2, 3]

Choose 3 values:
[2, 1, 3]

Each element can be selected only once.
This concept becomes important in sampling.
"""



# ============================================================
# 19. RANDOM SEED
# ============================================================

"""
Now we reach one of the MOST IMPORTANT concepts:
RANDOM SEED

Normally, random numbers change every time
you run your program.

But sometimes we want the SAME random numbers
every time.

This is called : REPRODUCIBILITY
"""




# ============================================================
# 20. np.random.seed()
# ============================================================

np.random.seed(42)
print(np.random.rand())

"""
If you run this with the same seed,
you will get the same random value.

42 is simply a commonly used example.
It has no special mathematical meaning.

NumPy Random Seed Cheat Sheet

1. What is randomness?
np.random.randint(1, 10)

Can produce different values each time:

5
2
8
...

So the output is not predictable.

2. What is a seed?
A seed fixes the starting point of the random number generator.
np.random.seed(42)
Now random results become reproducible.
np.random.seed(42)
print(np.random.randint(1, 10, 5))

Run the same code again with the same seed → same output.
[7 4 8 5 7]   ← example
Remember
Same seed → Same random sequence
Different seed → Different sequence
No seed → Usually different sequence each run

3. Why is this important in ML?
Suppose you randomly split data into training and testing sets.
Without a fixed seed:
Run 1 → different split
Run 2 → different split
Run 3 → different split

Your results can change.
With a seed:

Run 1 → same split
Run 2 → same split
Run 3 → same split
This is called reproducibility.

Important: Old vs Modern NumPy
Older style

np.random.seed(42)
numbers = np.random.randint(1, 100, 5)

Modern recommended style

rng = np.random.default_rng(42)
numbers = rng.integers(1, 100, size=5)

For your learning, learn both, but prefer the modern default_rng() style in new code.

Practice Problems
Problem 1
What happens?
np.random.seed(42)
print(np.random.randint(1, 10))
print(np.random.randint(1, 10))

Run it twice.

Question: Will the outputs be the same?
Yes. Same seed → same sequence.

Problem 2
np.random.seed(10)
a = np.random.randint(1, 100, 5)

np.random.seed(10)
b = np.random.randint(1, 100, 5)

print(a)
print(b)

Are a and b the same?
Yes.

Problem 3
np.random.seed(10)
a = np.random.randint(1, 100, 5)

np.random.seed(20)
b = np.random.randint(1, 100, 5)

Will a and b usually be the same?
No. Different seeds → different sequences.

Problem 4
rng = np.random.default_rng(42)

a = rng.integers(1, 10, size=5)
b = rng.integers(1, 10, size=5)

Will a and b be identical?
No. The generator continues from where it left off.
But if you create another generator with the same seed:

rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)

print(rng1.integers(1, 10, 5))
print(rng2.integers(1, 10, 5))

They produce the same sequence.
"""



# ============================================================
# 21. SAME SEED = SAME RANDOM SEQUENCE
# ============================================================

np.random.seed(42)
a = np.random.rand(5)
np.random.seed(42)
b = np.random.rand(5)

print(a) # [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
print(b) # [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
print(np.array_equal(a, b))

"""
Output : True

Why?
Because the same seed creates the same
pseudo-random sequence.
This is reproducibility.
"""


# ============================================================
# 22. DIFFERENT SEEDS
# ============================================================

np.random.seed(42)
a = np.random.rand(5)

np.random.seed(100)
b = np.random.rand(5)

print(a)
print(b)

"""
Different seeds produce different sequences.
"""


# ============================================================
# 23. WHY REPRODUCIBILITY MATTERS
# ============================================================


"""
Imagine you train an ML model.
You use random values during:
data splitting
initialization
sampling

Today you get:
Accuracy = 91%

Tomorrow you run the same code and get:
Accuracy = 88%

You may not know exactly why.

If randomness is controlled with a seed,
you can reproduce the experiment.

This makes:
debugging
testing
experiments
research

much easier.
"""



# ============================================================
# 24. MODERN NUMPY RANDOM GENERATOR
# ============================================================

"""
NumPy's modern random-number API uses : np.random.default_rng()
Example:
"""

rng = np.random.default_rng(42)
numbers = rng.random(5)
print(numbers)

"""
The number 42 is the seed.
This creates a Generator object.
Then we use the generator to create random values.
"""


# ============================================================    a 
# 25. rng.integers()
# ============================================================

"""
Modern equivalent of:
np.random.randint() is: rng.integers()
"""

rng = np.random.default_rng(42)
numbers = rng.integers(
1,
100,
size=10
)
print(numbers)

"""
Possible output : [ 9 77 65 44 43 86  8 70 20  9]

Important:
1 is included
100 is excluded

So values are : 1 to 99
"""



# ============================================================
# 26. rng.random()
# ============================================================


"""
Modern equivalent of:
np.random.rand() is : rng.random()
"""

rng = np.random.default_rng(42)
print(rng.random())

"""
Random floating-point number from:
0 <= number < 1
"""


# ============================================================
# 27. rng.random() MULTIPLE VALUES
# ============================================================

rng = np.random.default_rng(42)
numbers = rng.random(5)
print(numbers)

"""
5 random floating-point values.
All are in: [0, 1)
"""


# ============================================================
# 28. rng.random() 2D ARRAY
# ============================================================


rng = np.random.default_rng(42)
matrix = rng.random((2, 3))
print(matrix)

"""
Shape: (2, 3)
All values are between 0 and 1.
"""



# ============================================================
# 29. rng.normal()
# ============================================================

"""
Modern equivalent of : np.random.randn()
is : rng.normal()
But rng.normal() is more powerful because
you can specify:
mean
standard deviation
size
"""

rng = np.random.default_rng(42)
numbers = rng.normal(
loc=0,
scale=1,
size=5
)

print(numbers)
"""
loc: Mean
scale:Standard deviation
Here:
mean = 0
std  = 1

So this is a standard normal distribution.
"""


# ============================================================
# 30. NORMAL DISTRIBUTION WITH CUSTOM MEAN
# ============================================================


rng = np.random.default_rng(42)
numbers = rng.normal(
loc=100,
scale=10,
size=5
)

print(numbers)
"""
Now:
Mean = 100
Standard deviation = 10
So values will generally be around 100.
"""


# ============================================================
# 31. rng.choice()
# ============================================================

"""
Modern equivalent of : np.random.choice()
Example:
"""

rng = np.random.default_rng(42)
arr = np.array([
10,
20,
30,
40,
50
])

selected = rng.choice(arr, size=3)
print(selected)
# 3 random values are selected.


# ============================================================
# 32. rng.choice() WITHOUT REPLACEMENT
# ============================================================

rng = np.random.default_rng(42)
arr = np.array([
10,
20,
30,
40,
50
])

selected = rng.choice(
arr,
size=3,
replace=False
)
print(selected)

"""
No element can appear more than once.
"""


# ============================================================
# 33. MODERN GENERATOR REPRODUCIBILITY
# ============================================================

rng1 = np.random.default_rng(42)
a = rng1.random(5)
rng2 = np.random.default_rng(42)
b = rng2.random(5)

print(a)
print(b)

print(np.array_equal(a, b)) # True

"""
Same seed
↓
Same generator sequence
↓
Same results
This is reproducibility.
"""


# ============================================================
# 34. WHY default_rng() IS PREFERRED
# ============================================================

"""
For NEW NumPy code, the Generator approach is generally
preferred over relying on the older global random state.
Old style:
np.random.seed(42)
np.random.rand()
np.random.randint()

Modern style:
rng = np.random.default_rng(42)

rng.random()
rng.integers()
rng.normal()
rng.choice()

The Generator approach gives you a separate random
number generator object.

This makes random behavior easier to control,
especially in larger programs.
"""


# ============================================================
# 35. GLOBAL RANDOM STATE VS GENERATOR
# ============================================================

"""
OLD STYLE : np.random.seed(42)
The global NumPy random state is affected.
MODERN STYLE:
rng = np.random.default_rng(42)

The Generator object has its own state.
This is especially useful when different parts of
a program need independent random streams.
"""


# ============================================================
# 36. MULTIPLE GENERATORS
# ============================================================

rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(100)

print(rng1.random(3))
print(rng2.random(3))


"""
These are two separate generators.
They can produce independent sequences.
"""



# ============================================================
# 37. RANDOM INTEGER DATASET
# ============================================================


"""
We can create a small synthetic dataset using random numbers.
Suppose:
10 students
marks between 0 and 100
"""

rng = np.random.default_rng(42)
marks = rng.integers(
0,
101,
size=10
)

print(marks)

"""
Possible output:
[ 9 77 65 44 43 86  8 70 20  9]
101 is not used because the upper limit is excluded.
Therefore:
0 <= marks <= 100
"""


# ============================================================
# 38. RANDOM HEIGHT DATA
# ============================================================

"""
Suppose height is approximately normally distributed.
We can generate synthetic height data.
"""

rng = np.random.default_rng(42)
height = rng.normal(
loc=170,
scale=10,
size=10
)

print(height)

"""
Mean:
170 cm

Standard deviation:
10 cm

This is synthetic data.
It is NOT actual human data.
"""


# ============================================================
# 39. RANDOM TRAINING DATA
# ============================================================

rng = np.random.default_rng(42)
X = rng.random((100, 3))
print(X)
print(X.shape)

"""
We created:
100 samples
3 features
```

Shape: (100,3)
This is similar to the shape of a basic ML feature matrix.
"""


# ============================================================
# 40. RANDOM TARGET VALUES
# ============================================================

rng = np.random.default_rng(42)
X = rng.random((100, 3))
y = rng.integers(
0,
2,
size=100
)
print(X.shape)
print(y.shape)

"""
X: (100, 3)
100 samples
3 features

y:
(100,)
100 target values

Here:
0 and 1
could represent two classes.
This is a simple example of synthetic classification data.
"""


# ============================================================
# 41. RANDOM BINARY DATA
# ============================================================

rng = np.random.default_rng(42)
binary = rng.integers(
0,
2,
size=10
)
print(binary)
"""
Possible output:
[0 1 1 0 0 1 0 1 1 0]
Because:
low = 0
high = 2
possible values are:
0
1
"""


# ============================================================
# 42. RANDOM SAMPLING FROM AN ARRAY
# ============================================================

students = np.array([
"A",
"B",
"C",
"D",
"E"
])

rng = np.random.default_rng(42) # 42 is the seed => Same seed → reproducible selection.
selected = rng.choice(
students, # Randomly select from students.
size=2, # select two students
replace=False # Do not select the same student more than once.
)
print(selected) 


"""
Two different students were randomly selected.
This can represent sampling without replacement.
"""



# ============================================================
# 43. RANDOM SHUFFLING
# ============================================================


"""
Shuffling means randomly changing the order
of elements.
Modern Generator:
rng.shuffle()

"""
arr = np.array([1, 2, 3, 4, 5])
rng = np.random.default_rng(42)  # 42 is the seed, so the shuffle is reproducible.
rng.shuffle(arr) # This changes arr directly 
print(arr) # shows your actual shuffled array:


"""
The elements remain the same.
Only their ORDER changes.
"""



# ============================================================
# 44. IMPORTANT: shuffle() MODIFIES THE ARRAY
# ============================================================

arr = np.array([1, 2, 3, 4, 5])
rng = np.random.default_rng(42)
rng.shuffle(arr)
print(arr)

"""
shuffle() changes arr itself.
It does not create a separate shuffled array
that you need to assign back.
"""



# ============================================================
# 45. SHUFFLING IS IMPORTANT IN ML
# ============================================================



"""
Suppose your dataset is ordered like:
Class A
Class A
Class A
Class A
Class B
Class B
Class B
Class B
```

If you split it without considering the ordering,
you might create a poor train/test division.

Random shuffling can help mix the samples before
splitting the data.

Later, libraries such as scikit-learn provide
specialized tools for this.
"""




# ============================================================
# 46. RANDOM PERMUTATION
# ============================================================


"""
A permutation is a rearrangement of elements.
Generator:
rng.permutation()

"""
arr = np.array([1, 2, 3, 4, 5])
rng = np.random.default_rng(42)
result = rng.permutation(arr) # permutation() creates a shuffled copy => the original array stays unchanged
print(result)

"""
Unlike shuffle():
permutation() returns a new array.
The original arr remains unchanged.
"""


# ============================================================
# 47. shuffle() VS permutation()
# ============================================================

"""
shuffle(): changes the original array

Example:
rng.shuffle(arr)

permutation():
Returns a shuffled copy

Example :
shuffled = rng.permutation(arr)

MEMORY:
shuffle -> modifies original
permutation -> returns new array
"""


# ============================================================
# 48. RANDOM VALUES BETWEEN CUSTOM RANGE
# ============================================================
"""
rng.random() gives:
[0, 1)

If we want a different range,
we can transform the result.

Example:
random value between 10 and 20
"""

rng = np.random.default_rng(42)
random_value = rng.random()
value = 10 + random_value * 10
print(value)
"""
The result is between:
10
and
20

This idea is based on scaling a [0,1) random value.
"""


# ============================================================
# 49. RANDOM FLOATS IN A RANGE
# ============================================================

rng = np.random.default_rng(42)
values = rng.uniform(  # uniform means Uniform means every value in the range has approximately the same chance of being selected.
low=10,
high=20,
size=5 # Generate 5 numbers.
)
print(values) # looks like [17.73956049 14.3887844  18.5859792  16.97368029 10.94177348]

"""
rng.uniform() is a clearer choice when you directly
want random floating-point values in a specified range.
Range:
[10, 20)

rng.integers(10, 20, 5)  # random integers
rng.uniform(10, 20, 5)   # random decimals
""" 



# ============================================================
# 50. RANDOM NORMAL DATA => normal() → values concentrated around a mean
# ============================================================

rng = np.random.default_rng(42)
values = rng.normal(
loc=50,
scale=5,
size=10
)
print(values)
"""
➡️ Generate 10 numbers
So:
normal distribution => Most generated values tend to be near 50, while values farther away from 50 are less likely.
mean = 50
std = 5
10 values

This generates synthetic data around 50.
"""



# ============================================================
# 51. UNDERSTANDING DISTRIBUTION
# ============================================================

"""
A DISTRIBUTION describes how values are spread.
Two important distributions from this phase:

---

## UNIFORM DISTRIBUTION
Example:
rng.uniform(0, 1, size=1000) # Generate 1000 random decimal numbers between 0 and 1.
Values are spread across the range


## NORMAL DISTRIBUTION
Example
rng.normal(0, 1, size=1000) # Generate 1000 random numbers from a normal distribution with mean 0 and standard deviation 1.
Values tend to cluster around the mean.
This produces the familiar bell-shaped distribution.
You do not need advanced probability theory right now.
Just understand the basic difference.
"""


# ============================================================
# 52. SEED AND REPRODUCIBILITY EXAMPLE
# ============================================================

"""
Suppose we generate fake exam marks.
Without a fixed seed:
Results can change each time.
With a fixed seed:
Results can be reproduced.

"""
rng = np.random.default_rng(42) # creates a generator starting from seed 42.
marks1 = rng.integers(
0,
101,
size=10
) # generates 10 random integers from 0 to 100.
rng = np.random.default_rng(42) # So it starts from exactly the same random state.
marks2 = rng.integers(
0,
101,
size=10
) # produces the same 10 numbers.

print(marks1)
print(marks2)
print(np.array_equal(marks1, marks2)) # True

"""
Output:
True
"""



# ============================================================
# 53. ONE GENERATOR, MULTIPLE OPERATIONS
# ============================================================

rng = np.random.default_rng(42)
print(rng.random(3)) # 3 random decimals between 0 and 1.
print(rng.integers(1, 10, size=3)) # Generates 3 random integers from 1 to 9    
print(rng.normal(0, 1, size=3)) # Generates 3 random numbers from a normal distribution: mean =0, sd = 1

"""
The SAME generator can produce different types of
random data.
This is one of the benefits of the Generator API.
"""


# ============================================================
# 54. RANDOM CHOICE WITH PROBABILITIES
# ============================================================

"""
choice() can also assign probabilities.
Example:
A -> 70%
B -> 20%
C -> 10%
"""

rng = np.random.default_rng(42)
values = np.array(["A", "B", "C"])
selected = rng.choice(
values, # -> what to choose from
size=10, # -> how many to choose => Select 10 values. => Since you didn't specify replace=False, the default is: replace=True
p=[0.7, 0.2, 0.1]    # => here p means The probabilities correspond to A, B, C in the same order:
)
print(selected) # ex : A A A A A A A B B C / A A A A A A A A A A

"""
p specifies the probability of selecting each value.
The probabilities must add up to 1.
Here:
0.7 + 0.2 + 0.1 = 1.0
"""



# ============================================================
# 55. RANDOM SAMPLE FROM A DATASET
# ============================================================

data = np.array([
[1, 10],
[2, 20],
[3, 30],
[4, 40],
[5, 50]
])

rng = np.random.default_rng(42) # # Create random number generator with seed 42
indices = rng.choice(  #  Randomly select 2 row indices
len(data), # Number of rows in data (5)
size=2,    # select two rows
replace=False  # dont select the same row twice
) 
sample = data[indices]  # for example indices might contain [0 3]
print(sample) # Then you can actually get those rows


"""
We:
1. Generated random row indices.
2. Used those indices to select rows.
This is a basic form of random sampling.
"""




# ============================================================
# 56. RANDOM TRAIN/TEST IDEA
# ============================================================

"""
Suppose we have 10 samples.
We want:
8 training samples
2 testing samples

One basic idea is:

1. Create random indices.
2. Shuffle them.
3. Take the first 8 for training.
4. Take the remaining 2 for testing.

Example:
"""
# doing a random train-test split using indices
data = np.arange(10) # creates [0 1 2 3 4 5 6 7 8 9]
rng = np.random.default_rng(42) # 42 = seed, so the random order is reproducible.
indices = rng.permutation(len(data)) # len(data) = 10 => randomly rearranges the numbers 0 to 9.    
train_indices = indices[:8] # take first 8 indices => ex : [5 6 0 7 3 2 4 9]
test_indices = indices[8:] # Takes everything from position 8 onward: => ex : [1 8]
print("Train indices:")
print(train_indices) # [5 6 0 7 3 2 4 9]
print("Test indices:")
print(test_indices) # [1 8]


"""
This demonstrates the basic idea behind random
data splitting.

In real projects, use dedicated ML tools such as
scikit-learn for robust train/test splitting.
"""



# ============================================================
# 57. RANDOM INITIALIZATION IN ML
# ============================================================


"""
Random initialization is very important in neural networks.
A neural network contains parameters called:
weights
biases

These parameters often need to start with
small random values.
For example:
weights = rng.normal(
    0,
    0.01,
    size=(3, 4)
)

This creates a:
3 x 4

matrix of small random values.
Later, when you study neural networks,
you will learn why initialization matters.
"""



# ============================================================
# 58. SIMPLE WEIGHT INITIALIZATION EXAMPLE
# ============================================================

rng = np.random.default_rng(42)
weights = rng.normal(
loc=0,
scale=0.01,
size=(3, 4)
)
print(weights)

"""
Shape:
(3, 4)

The weights are small random values centered around 0.
This is only a basic demonstration.
Actual neural-network initialization methods
can be more sophisticated.
"""


# ============================================================
# 59. RANDOMNESS IN EXPERIMENTS
# ============================================================

"""
Suppose you are testing an algorithm.
Experiment 1:
Random data
-> algorithm
-> result

Experiment 2:
Random data
-> algorithm
-> result

If the random data changes every time,
your results may also change.

Using a fixed seed allows you to reproduce
the same experiment.
This is extremely useful during development.
"""



# ============================================================
# 60. RANDOMNESS DOES NOT MEAN TRUE RANDOMNESS
# ============================================================

"""
NumPy's normal random-number generators produce
PSEUDO-RANDOM numbers.
They are generated algorithmically.
If you know the generator state/seed,
the sequence can be reproduced.
Therefore:

same seed
    ↓
same sequence
    ↓
reproducible experiment

For cryptographic security, NumPy's ordinary
random generators are NOT the correct tool.
"""



# ============================================================
# 61. OLD STYLE QUICK REFERENCE
# ============================================================

"""
OLD / COMMON NUMPY RANDOM API:

np.random.rand()
    -> uniform random floats in [0, 1)

np.random.randn()
    -> standard normal random values

np.random.randint()
    -> random integers

np.random.choice()
    -> random selection

np.random.seed()
    -> controls the global random state
"""



# ============================================================
# 62. MODERN GENERATOR QUICK REFERENCE
# ============================================================


"""
MODERN GENERATOR API:
rng = np.random.default_rng(42)
rng.random()
    -> uniform random floats in [0, 1)

rng.integers()
    -> random integers

rng.normal()
    -> normal-distribution values

rng.choice()
    -> random selection

rng.shuffle()
    -> shuffle an array in place

rng.permutation()
    -> return a shuffled copy

rng.uniform()
    -> uniform random floats in a specified range
"""


# ============================================================
# 63. OLD VS MODERN
# ============================================================

"""
OLD:
np.random.seed(42)
numbers = np.random.randint(
    1,
    100,
    size=10
)

MODERN:
rng = np.random.default_rng(42)
numbers = rng.integers(
    1,
    100,
    size=10
)

For new code:
Prefer the modern Generator approach.
"""


# ============================================================
# 64. IMPORTANT RANGE RULE
# ============================================================

"""
Remember:
randint()
integers()
use:
low = INCLUDED
high = EXCLUDED

Example:
rng.integers(1, 5)

Possible values:
1
2
3
4

NOT:
5

For random floats:
rng.random()

gives:
0 <= x < 1
"""


# ============================================================
# 65. COMPLETE EXAMPLE
# ============================================================

"""
Let's create a small synthetic ML dataset.
"""
rng = np.random.default_rng(42) # Creates a random generator with seed 42, so results are reproducible.
# 100 samples and 3 features.
X = rng.normal(
loc=0,   # => mean = 0
scale=1, # standard deviation = 1
size=(100, 3) # size = 100 rows X 3 columns
)
# 100 binary target values.
y = rng.integers(
0, # => minimum
2, # => upper limit not included
size=100  # generate 100 values only 0's and 1's
) 
# example : [1 0 1 1 0 0 1 ...]
# X->features | y-> labels
# X.shape = (100, 3)| y.shape = (100,)
print("X:")
print(X)
print()
print("X shape:")
print(X.shape)
print()
print("y:")
print(y)
print()
print("y shape:")
print(y.shape)

"""
X shape:(100, 3)
Meaning:
100 samples
3 features

y shape:
(100,)

Meaning:
100 target values

This is a very basic example of how random numbers
can be used to create synthetic ML data.
"""



# ============================================================
# 66. FINAL MEMORY MAP
# ============================================================

"""
RANDOM NUMBERS
|
-------------------------------
|              |              |
FLOAT          INTEGER        CHOICE
|              |              |
random()       integers()     choice()
|              |
----------------    |
|              |    |
uniform         normal |
|              |    |
random()       normal() |
|
RANDOM DATA
|
-------------------------
|           |           |
sampling    experiments    ML
|
reproducibility
|
SEED
|
default_rng(42)
"""



# ============================================================
# 67. MOST IMPORTANT CONCEPT
# ============================================================

"""
SEED
----

A seed controls the starting point of a
pseudo-random sequence.

Example:

rng = np.random.default_rng(42)
If you recreate the generator with the SAME seed:

rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)

then they produce the same sequence when used
in the same way.

Therefore:

SAME SEED
    ↓
SAME RANDOM SEQUENCE
    ↓
REPRODUCIBLE RESULTS

"""


# ============================================================
# 68. PRACTICE QUESTIONS
# ============================================================

"""
QUESTION 1
----------
Generate one random floating-point number
between 0 and 1 using:

```
np.random.rand()
```

QUESTION 2
----------
Generate 10 random floating-point numbers
between 0 and 1.


QUESTION 3
----------
Generate a 3 x 4 random matrix using rand().


QUESTION 4
----------
Generate 5 random numbers using randn().
"""

"""
QUESTION 5
----------
Generate 10 random integers between 1 and 50.
"""

"""
QUESTION 6
----------
Create:
arr = np.array([10, 20, 30, 40, 50])
Randomly select 2 values without replacement.

QUESTION 7
----------
Create a Generator using seed 42.
Then generate:
10 random integers
from 1 to 100

QUESTION 8
----------
Create a 5 x 3 array of random values
from a standard normal distribution.

QUESTION 9
----------
Generate 10 values from a normal distribution
with:
mean = 100
standard deviation = 15

QUESTION 10
-----------
Create:
arr = np.array([1, 2, 3, 4, 5])
Shuffle the array using a Generator.

QUESTION 11
-----------
Create the same random sequence twice
using the same seed.
Check whether they are equal using:
np.array_equal()

QUESTION 12
-----------
Create:
100 samples
5 features

using random normal values.
What should the shape be?

QUESTION 13
-----------
Create 100 binary target values:
0
1
using rng.integers().

QUESTION 14
-----------
Explain the difference between:
rng.random()
and:
rng.normal()

QUESTION 15
-----------
ransom seed

"""


# ============================================================
# 69. FINAL PHASE 12 SUMMARY
# ============================================================


"""
YOU SHOULD NOW UNDERSTAND:

1. RANDOMNESS
   Random numbers are useful for:
   sampling
   experiments
   simulations
   ML

2. np.random.rand()
   Random floats in [0, 1)

3. np.random.randn()
   Random values from standard normal distribution

4. np.random.randint()
   Random integers

5. np.random.choice()
   Randomly select elements

6. SEED
   Controls reproducibility of pseudo-random sequences

7. default_rng()
   Modern NumPy random-number generator

8. rng.random()
   Random floats in [0, 1)

9. rng.integers()
   Random integers

10. rng.normal()
    Random normal-distribution values

11. rng.choice()
    Random sampling

12. rng.shuffle()
    Shuffle an array in place

13. rng.permutation()
    Return a shuffled copy

14. rng.uniform()
    Random floats from a specified uniform range

MOST IMPORTANT:
For NEW NumPy code, remember:
rng = np.random.default_rng(42)

Then use:
rng.random()
rng.integers()
rng.normal()
rng.choice()
rng.shuffle()
rng.permutation()

ML CONNECTION:

Random numbers
      ↓
Random sampling
      ↓
Data splitting
      ↓
Experiments
      ↓
Reproducibility
      ↓
Model initialization
      ↓
Neural Networks
      ↓
Machine Learning

The key idea is NOT memorizing every random function.
The key idea is understanding:

RANDOMNESS
    +
DISTRIBUTIONS
    +
SEEDS
    +
REPRODUCIBILITY
    +
SHAPES

Once these are clear, you are ready to move forward.
"""

# Modern NumPy style
rng = np.random.default_rng(42)

# Random floats: [0, 1)
rng.random(5)

# Random integers: 1 to 99
rng.integers(1, 100, size=10)

# Normal distribution
rng.normal(loc=0, scale=1, size=10)

# Random selection
rng.choice(arr, size=3, replace=False)

# Shuffle original array
rng.shuffle(arr)

# Return shuffled copy
rng.permutation(arr)