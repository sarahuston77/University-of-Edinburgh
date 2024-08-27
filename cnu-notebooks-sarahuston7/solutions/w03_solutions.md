# Week 3 solutions


## Exercise 1

```python
import numpy as np

n = np.random.randint(1, 1001)

if n % 21 == 0:
    # if n is a multiple of 3x7, it's a multiple of both 3 and 7
    print(n, 'is a multiple of both 3 and 7.')
elif n % 3 == 0 or n % 7 == 0:
    # not a multiple of both, but a multiple of either
    print(n, 'is a multiple of one of 3 or 7.')
else:
    # the last possible case: not a multiple of either
    print(n, 'is not a multiple of 3 nor 7.')
```

---

## Exercise 2

```python
zen = 'If the implementation is hard to explain, it is a bad idea. If the implementation is easy to explain, it may be a good idea.'
count = 0

# The .split() method returns a list of words
for word in zen.split():
    if 'e' in word:
        print(word)
    elif 'i' in word:
        print(word[0])
    else:
        count += 1
```

---

## Exercise 3

The outer loop runs for 10 iterations, and at each of these, the inner loop runs for 5 iterations. If the conditional `break` was absent, then we should see that `count` has reached $5\times 10 = 50$ after the loops: it is incremented 5 times each of the 10 iterations of the outer loop. The `print()` statement is inside the outer loop, but outside the inner loop: it executes only once per iteration of the outer loop, so we should see a total of 10 numbers displayed below the cell.

When `count` reaches 18, the **inner** loop breaks (but not the outer loop!). `count` is then printed, with the inner loop only having completed 3 iterations this time. We then start a new iteration of the outer loop, and enter the inner loop again (with `j` set to `0`), where `count` is incremented by 1 (so now `count` is `19`). Since `count > 17` is still `True`, we break the inner loop again (after just 1 iteration), `print(count)`, and start the new iteration of the outer loop. This continues until we've completed the 10 iterations of the outer loop.

`break` only breaks the **innermost** loop.

```python
count = 0

for i in range(10):
    for j in range(5):
        count += 1
        if count > 17:
            break   # this break...
    # ... leads us exactly here (just outside of the inner loop)
    print(count)
```

---

## Exercise 4

```python
import numpy as np

# Create the matrix M
M = np.array([[9, 3, 0], [-2, -2, 1], [0, -1, 1]])

# Create the vector y
y = np.array([0.4, -3, -0.3])
```

## Exercise 5

```python
import numpy as np

def dot_prod(u, v):
    '''
    Returns the dot product of vectors u and v.
    '''
    return np.sum(u * v)
```

---

## Exercise 6

```python
n = 4

# Initialise A with zeros
A = np.zeros([n, n])

# Loop over the rows...
for i in range(n):
    # Loop over the columns...
    for j in range(n):
        if i < j:
            A[i, j] = i + 2*j
        else:
            A[i, j] = i * j

print(A)
```

---

## Exercise 7

```python
def mat_power(A, n):
    '''
    Returns the nth power of the square matrix A.
    '''
    An = A.copy()

    # Use a loop to keep multiplying
    for i in range(1, n):
        An = An @ A

    return An

# Test our function on the test matrix A
A = np.array([[0, 1], [-1, 0]])

for n in range(1, 13):
    print(f'A^{n} = ')
    print(mat_power(A, n))
```

When $n$ is divisible by $4$, $A^n$ is the identity matrix. The matrix $A$ is a fourth root of the $2 \times 2$ identity matrix.
