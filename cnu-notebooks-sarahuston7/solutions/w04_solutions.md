# Week 4 solutions

## Exercise 1

```python
import numpy as np

# create the matrix m
m = np.array([[9, 3, 0], [-2, -2, 1], [0, -1, 1]])

# create the vector y
y = np.array([0.4, -3, -0.3])

# solve the system, display the solution
x = np.linalg.solve(m, y)
print(x)
```

---

## Exercise 2

Using a `for` loop:

```python
import numpy as np

def linsolve_diag(A, b):
    '''
    Solves the diagonal system Ax = b for x,
    assuming A is invertible.
    '''
    n = b.shape
    x = np.zeros(n)
    for i in range(n[0]):
        x[i] = b[i] / A[i, i]
    return x


# Expected solution: [20, 10]
A = np.array([[2, 0],
              [0, 0.5]])
b = np.array([40, 5])
print(linsolve_diag(A, b))
```

--

Using the `np.diag()` function:

```python
import numpy as np

def linsolve_diag(A, b):
    '''
    Solves the diagonal system Ax = b for x,
    assuming A is invertible.
    '''
    x = b / np.diag(A)
    return x


# Expected solution: [20, 10]
A = np.array([[2, 0],
              [0, 0.5]])
b = np.array([40, 5])
print(linsolve_diag(A, b))
```

---

## Exercise 3

We can also use `np.diag()` to construct a diagonal matrix:

```python
# Construct A by first constructing the diagonal as a vector
a = np.array([3, -1, 10])
A = np.diag(a)

# Construct b
b = np.array([3, 1, 1])

# Solve the system and print the result
x = linsolve_diag(A, b)
print(x)
```

---

## Exercise 4

```python
import time

# Create a randomised invertible matrix A and vector b
N = 1000
A = np.diag(np.random.random([N])) + np.eye(N)
b = np.random.random([N])

# Use both methods to solve the system, time them
t0 = time.time()
x = np.linalg.solve(A, b)
t1 = time.time()
x = linsolve_diag(A, b)
t2 = time.time()

# Display the results
print(f"N = {N:d}")
print(f"Time taken by np.linalg.solve: {t1 - t0:.6f} seconds")
print(f"Time taken by linsolve_diag: {t2 - t1:.6f} seconds")
```

Here `linsolve_diag()` is provided with much more information about the input matrix (that it is diagonal), and hence is more efficient than `np.linalg.solve()`, as it completely avoids any unnecessary calculations.

---

## Exercise 5

```python
def linsolve_lt(A, b):
    '''
    Solves the lower triangular system Ax = b.
    '''
    # Find the size of the system
    N = b.shape[0]

    # Initialise an array x of the correct size, full of zeros
    x = np.zeros(N)

    # Loop over the rows of the system, top to bottom
    for i in range(N):
        # Rearrange the equation (Ax)_i = b_i to make x_i the subject
        x[i] = (b[i] - A[i, :i] @ x[:i]) / A[i, i]
    return x

# Solving the system in the example above
A = np.array([[2, 0, 0],
              [-1, 1, 0],
              [-1, 1, 2]], dtype=float)
b = np.array([4, 1, 4], dtype=float)
x = linsolve_lt(A, b)
print(x)
```

The ith element of the vector $Ax$ is given in general by

$$
(Ax)_i = \sum_{j=1}^n a_{ij} x_j,
$$

the dot product between $x$ and the ith row of $A$. Here, since the matrix is lower triangular, we have $a_{ij} = 0$ when $j > i$, so the dot product reduces to

$$
(Ax)_i = \sum_{j=1}^i a_{ij} x_j.
$$

With forward substitution, we assume that when calculating $x_i$, the previous elements $x_j, j = 1, \dots, i-1$ are already known, so we can rearrange the ith equation $(Ax)_i = b_i$ to make $x_i$ the subject:

$$
x_i = \frac{1}{a_{ii}} \left(b_i - \sum_{j=1}^{i-1} a_{ij} x_j \right).
$$

---

## Exercise 6

```python
def linsolve_ut(A, b):
    '''
    Solves the upper triangular system Ax = b.
    '''
    # Find the size of the system
    N = b.shape[0]

    # Initialise an array x of the correct size, full of zeros
    x = np.zeros(N)

    # Loop over the rows of the system, bottom to top
    for i in range(N-1, -1, -1):
        # Rearrange the equation (Ax)_i = b_i to make x_i the subject
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
    
    return x

# Testing with an example
A = np.array([[1, 1],
              [0, 0.5]])
b = np.array([1, 1])
x = linsolve_ut(A, b)
print(x)
```

---

## Exercise 7

```python
import time

# Create a randomised invertible upper triangular matrix A and vector b
N = 1000
A = np.triu(np.random.random([N])) + np.eye(N)
b = np.random.random([N])

# Test the 2 methods and time them
t0 = time.time()
x = np.linalg.solve(A, b)
t1 = time.time()
x = linsolve_lt(A, b)
t2 = time.time()

# Display the results
print(f"N = {N:d}")
print(f"Time taken by np.linalg.solve: {t1 - t0:.6f} seconds")
print(f"Time taken by linsolve_ut: {t2 - t1:.6f} seconds")
```

Here, as was the case for `linsolve_diag()` earlier, `linsolve_lt()` is provided with more information about the input matrix (that it is lower triangular), and hence is more efficient than `np.linalg.solve()` (as it avoids unnecessary computations e.g. multiplications by zero).

---

## Exercise 8

```python
def row_op(A, alpha, i, beta, j):
    '''
    Applies row operation beta*A_j + alpha*A_i to A_j,
    the jth row of the matrix A.
    Changes A in place.
    '''
    # Apply the row operation on the jth row of A
    A[j, :] = beta * A[j, :] + alpha * A[i, :]

A = np.array([[2, 0],
              [1, 2]])
alpha, beta = 2, -1
i, j = 1, 0
print(row_op(A, alpha, i, beta, j))
```

---

## Exercise 9

```python
def REF(A, b):
    '''
    Reduces the augmented matrix (A|b) into
    row echelon form, returns (C|d).
    '''
    # Build the augmented matrix
    N = A.shape[0]
    Aug = np.zeros([N, N+1])
    Aug[:, :N] = A
    Aug[:, N] = b
    
    # Loop over the columns
    for col in range(N-1):
        
        # In each column, loop over the rows below the diagonal
        for row in range(col+1, N):
            
            # Calculate alpha as -(leading element / diagonal element)
            alpha = -Aug[row, col] / Aug[col, col]
            
            # Perform the row operation in-place (beta is always 1 here)
            row_op(Aug, alpha, col, 1, row)
    
    # Split the result into C, d
    C = Aug[:, :N]
    d = Aug[:, N]
    return C, d

# Testing with an example
A = np.array([[1, 1, 1],
              [2, 1, -1],
              [1, 1, 2]], dtype=float)
b = np.array([2, 1, 0], dtype=float)

C, d = REF(A, b)
print(C)
print(d)
```

---

## Exercise 10

```python
def gauss(A, b):
    '''
    Solve the linear system Ax = b, given a square
    invertible matrix A and a vector b, using Gaussian elimination.
    '''
    # First step: reduce to row echelon form
    C, d = REF(A, b)
    
    # Second step: solve the resulting upper triangular system
    x = linsolve_ut(C, d)
    
    # Return the result
    return x


# Test the function
A = np.array([[1, 1, 1],
              [2, 1, -1],
              [1, 1, 2]], dtype=float)
b = np.array([2, 1, 0], dtype=float)

x = gauss(A, b)
print(x)

# Arbitrary test: make a random (probably invertible) matrix and vector
N = 20
A = np.random.random([N, N])
b = np.random.random(N)

# Solve for x
x = gauss(A, b)

# Ax - b should be, in theory, the zero vector
print(np.linalg.norm(A@x - b))
```
