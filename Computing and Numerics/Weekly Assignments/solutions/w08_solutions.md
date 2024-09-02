# Week 8 tutorial solutions

---

## Exercise 1

```python
import numpy as np

def quad_roots(a, b, c):
    '''
    Returns the roots of the polynomial
    ax^2 + bx + c, either real or complex.
    We assume a != 0.
    '''
    # Calculate the determinant
    delta = b**2 - 4*a*c
    
    # Choose depending on sign of delta
    if delta >= 0:
        return (-b - np.sqrt(delta)) / (2*a), (-b + np.sqrt(delta)) / (2*a)
    else:
        return (-b - 1j*np.sqrt(-delta)) / (2*a), (-b + 1j*np.sqrt(-delta)) / (2*a)


# Some quick testing...
print(quad_roots(1, 0, 1))     # roots of x^2 + 1 are -i, i
print(quad_roots(1, 0, -1))    # roots of x^2 - 1 are -1, 1
print(quad_roots(1, -2, 1))    # roots of x^2 - 2x + 1 are 1, 1
print(quad_roots(1, -2, 2))    # roots of x^2 - 2x + 2 are (1-i), (1+i)
```

---

## Exercise 2

```python
import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return np.sin(2.0 * np.pi * x) * np.exp(4.0 * x) + x

fig, ax = plt.subplots()

# Create an x-axis to plot the function
x_plot = np.linspace(0.0, 1.0, 1000)

# We draw a horizontal line between the first and last points,
# at y = 0, to represent the x-axis, in grey
ax.plot([x_plot[0], x_plot[-1]], [0.0, 0.0], color=[0.7, 0.7, 0.7])

# Now, we plot the function
ax.plot(x_plot, F(x_plot), "k-")
ax.set_xlim([x_plot[0], x_plot[-1]])
ax.set(xlabel=r"$x$", ylabel=r"$F \left( x \right)$")
plt.show()

# Check that F(0.2) and F(0.6) have different signs
print(F(0.2) * F(0.6) < 0)
```

There are three roots: a trivial root $x = 0$, a root near $x = 0.5$, and a root near $x = 1$. The product $F(0.2) \times F(0.6)$ is negative, indicating that $F \left( 0.2 \right)$ and $F \left( 0.6 \right)$ have opposite signs.


---

## Exercise 3

```python
# Initial interval and midpoint
a = 0.2
b = 0.6
c = 0.5 * (a + b)

# Iteration counter and tolerance
its = 0
tol = 1e-10

# Loop until the root is found
while abs(F(c)) >= tol:
    # Increment the iteration count
    its += 1

    if F(a) * F(c) <= 0.0:    # F(a) and F(c) have different signs (or one or both is zero) ...
        b = c                 # ... a root is between a and c (or equals a or c)
    else:
        a = c                 # Else, a root is between c and b (or equals b)
        
    # Find the next midpoint
    c = 0.5 * (a + b)

x_star = c
print(f"Root = {x_star}")
print(f"Number of iterations = {its}")
```

---

## Exercise 4

```python
def bisection(F, a, b, tol):
    '''
    Finds the root of F in the interval [a, b] using
    the bisection method, to within an error of tol.
    '''
    # Iteration count
    its = 0

    # Midpoint
    c = 0.5 * (a + b)
    
    # Loop until the root is found
    while abs(F(c)) >= tol:
        # Increment the iteration count
        its += 1

        if F(a) * F(c) <= 0.0:    # F(a) and F(c) have different signs (or one or both is zero) ...
            b = c                 # ... a root is between a and c (or equals a or c)
        else:
            a = c                 # Else, a root is between c and b (or equals b)

        # Find the next midpoint
        c = 0.5 * (a + b)
    
    # Return the root and the number of iterations
    return c, its


# Initial interval
a = 0.8
b = 1.2

# Tolerance
tol = 1e-10

# Compute the root
x_star, its = bisection(F, a, b, tol)
print(f"Root = {x_star}")
print(f"Number of iterations = {its}")
```

---

## Exercise 5

The slope of the line passing through the points $(a, F(a))$ and $(b, F(b))$ is given by

$$
\alpha = \frac{F(b) - F(a)}{b - a}.
$$

As the line passes through the point $(a, F(a))$, its equation is given by

$$
y - F(a) = \frac{F(b) - F(a)}{b - a} (x - a).
$$

We seek $c$ such that the line passes through the point $(c, 0)$. $c$ must satisfy the line equation, therefore

$$
0 - F(a) = \frac{F(b) - F(a)}{b - a} (c - a).
$$

Rearranging to find an expression for $c$ gives

$$
c = a - \frac{b - a}{F(b) - F(a)} F(a)
= \frac{a F(b) - a F(a) - b F(a) + a F(a)}{F(b) - F(a)}
= \frac{a F(b) - b F(a)}{F(b) - F(a)}.
$$

---

## Exercise 6

```python
def regula_falsi(F, a, b, tol):
    '''
    Finds the root of F in the interval [a, b] using
    the regula falsi method, to within an error of tol.
    '''
    # Iteration count
    its = 0
    
    # Initial x-intercept
    c = (a * F(b) - b * F(a)) / (F(b) - F(a))
    
    # Loop until the root is found
    while abs(F(c)) >= tol:
        # Increment the iteration count
        its += 1

        if F(a) * F(c) <= 0.0:    # F(a) and F(c) have different signs (or one or both is zero) ...
            b = c                 # ... a root is between a and c (or equals a or c)
        else:
            a = c                 # Else, a root is between c and b (or equals b)
            
        # Find the next x-intercept c
        c = (a * F(b) - b * F(a)) / (F(b) - F(a))
    
    # Return the root and the number of iterations
    return c, its


# Initial interval
a = 0.2
b = 0.6

# Tolerance
tol = 1e-10

# Compute the root
x_star, its = regula_falsi(F, a, b, tol)
print(f"Root = {x_star}")
print(f"Number of iterations = {its}")
```

---

## Exercise 7

```python
def regula_falsi_error(F, a, b, tol):
    '''
    Finds the root of F in the interval [a, b] using
    the regula falsi method, to within an error of tol.
    '''
    # Iteration count
    its = 0
    
    # Initial x-intercept
    c = (a * F(b) - b * F(a)) / (F(b) - F(a))
    
    # Store all guesses
    x = [c]
    
    # Loop until the root is found
    while abs(F(c)) >= tol:
        # Increment the iteration count
        its += 1

        if F(a) * F(c) <= 0.0:    # F(a) and F(c) have different signs (or one or both is zero) ...
            b = c                 # ... a root is between a and c (or equals a or c)
        else:
            a = c                 # Else, a root is between c and b (or equals b)
            
        # Find the next x-intercept c
        c = (a * F(b) - b * F(a)) / (F(b) - F(a))
        x.append(c)
    
    # Return the root and the number of iterations
    return np.array(x), its


# Initial interval
a = 0.2
b = 0.6

# Tolerance
tol = 1e-10

# Run the algorithm, compute the error between all guesses
# and the final solution
x_rf, its = regula_falsi_error(F, a, b, tol)
e_rf = np.abs(x_rf[:-1] - x_rf[-1])

fig, ax = plt.subplots(2, 1)

for p in [1, 1.5, 2, 2.5]:
    # Plot successive error ratios
    ax[0].plot(e_rf[1:] / e_rf[:-1]**p, marker='+', label=f'p = {p}')

ax[0].set(xlabel='Iteration', ylabel=r'$\frac{|e_{k+1}|}{|e_{k}|^p}$', ylim=[0., 1000.])
ax[0].legend()

# The ratios become very large very quickly for p=1.5 and above,
# clearly they are not constant. Zoom in on p=1 to check
ax[1].plot(e_rf[1:] / e_rf[:-1], marker='+')
ax[1].set(xlabel='Iteration', ylabel=r'$\frac{|e_{k+1}|}{|e_{k}|}$')

plt.show()
```

The ratio of errors is clearly constant for $p=1$ for most of the iterations, suggesting that the method is first order convergent; we seem to have $\alpha = 0.25$, which means that the guess improves by a factor of 4 at each iteration. For the first few iterations, we are not in a sufficiently close neighbourhood of the root, therefore we cannot yet observe the order of convergence. For the last few iterations, we are reaching very small values of the error, and we may be encountering roundoff errors due to the use of floating point numbers, which would explain the departure from the constant ratio of errors seen so far.
