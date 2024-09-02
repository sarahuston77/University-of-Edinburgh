# Week 10 solutions

---

## Exercise 1

```python
import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return x + 1 + np.arctan(10*x)

def Fp(x):
    return 1 + 10 / (1 + 100*x**2)

# Plot the function
x = np.linspace(-3, 3, 200)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].axhline(0, color=[0.7, 0.7, 0.7])
ax[0].plot(x, F(x))
ax[0].set(title='F(x)', xlabel='x', ylabel='F(x)')

# Newton's method
xk = [-2]
kmax = 30
for i in range(kmax):
    xk.append(xk[-1] - F(xk[-1]) / Fp(xk[-1]))

ax[1].plot(range(kmax + 1), xk, 'ro-')
ax[1].set(title='Newton iterations', xlabel='Iteration number', ylabel='Current guess')

plt.show()
```
For choices of $x_0$ too far from the root, we get stuck in a periodic cycle between 2 values on either side of the root.

---

## Exercise 2

```python
# Starting interval
a, b = -3., 3.

# Tolerance for bisection
tol = 1e-1

# Iteration count
k = 0

# Bisection method to refine the initial guess.
# Midpoint
c = 0.5 * (a + b)
xk = [c]

# Loop until the root is found
while abs(F(c)) >= tol:
    # Increment the iteration count
    k += 1

    if F(a) * F(c) <= 0.0:    # F(a) and F(c) have different signs (or one or both is zero) ...
        b = c                 # ... a root is between a and c (or equals a or c)
    else:
        a = c                 # Else, a root is between c and b (or equals b)

    # Find the next midpoint
    c = 0.5 * (a + b)
    xk.append(c)

# Mark the end of bisection iterations for plotting later
k_bisection = k

# Newton's method to refine the guess
tol = 1e-12

while abs(F(xk[-1])) >= tol:
    xk.append(xk[-1] - F(xk[-1]) / Fp(xk[-1]))
    k += 1
    
# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax[0].plot(range(k_bisection + 1), xk[:k_bisection + 1], 'ro', label='Bisection')
ax[0].plot(range(k_bisection + 1, k + 1), xk[k_bisection + 1:], 'bs', label='Newton')
ax[0].set(title='Bisection + Newton iterations', xlabel='Iteration number', ylabel='Current guess')

ax[1].plot(range(k_bisection + 1), abs(xk[:k_bisection + 1] - xk[-1]), 'ro', label='Bisection')
ax[1].plot(range(k_bisection + 1, k), abs(xk[k_bisection + 1:-1] - xk[-1]), 'bs', label='Newton')
ax[1].set(title='Bisection + Newton error', xlabel='Iteration number', ylabel='Absolute error', ylim=[-0.1, 0.3])

plt.show()
```

---

## Exercise 3

```python
def Jac(x1, x2):
    J = np.zeros([2, 2])
    J[0, 0] = 2 * x1
    J[0, 1] = 2 * x2
    J[1, 0] = -10 * x1 * x2
    J[1, 1] = 3 * x2**2 - 5 * x1**2
    return J


# Tests
assert np.allclose(Jac(0, 0), np.zeros([2, 2]))
assert np.allclose(Jac(1, 1), np.array([[2, 2], [-10, -2]]))
assert np.allclose(Jac(0, 1), np.array([[0, 2], [0, 3]]))
```

---

## Exercise 4

```python
def F(x1, x2):
    return np.array([F1(x1, x2), F2(x1, x2)])

# Initial guesses
x0 = np.array([[-3., 0.],
               [-2., 2.],
               [2., 2.],
               [3., 0.],
               [1., -3.],
               [-1., -3.]])

# Tolerance
tol = 1e-12

# Initialise an array to store all the roots
roots = []

# Loop over initial guesses to find all the roots
for x in x0:
    
    # Newton's method
    while np.linalg.norm(F(x[0], x[1])) >= tol:
        
        # Newton iteration
        e = -np.linalg.solve(Jac(x[0], x[1]), F(x[0], x[1]))
        x += e
        
    # Store the results
    roots.append(x)

# Plot the roots on the same graph
roots = np.array(roots)

fig, ax = plt.subplots(figsize=(8, 8))
ax.contour(X, Y, F1(X, Y), 0, colors='r')
ax.contour(X, Y, F2(X, Y), 0, colors='b')

ax.plot(roots[:, 0], roots[:, 1], 'go', markersize=10)

ax.set(xlabel='x1', ylabel='x2', xlim=[xmin, xmax], ylim=[ymin, ymax])
plt.show()
```

---

## Exercise 5

```python
import numpy as np
import matplotlib.pyplot as plt

def circle_coords(c, r):
    '''
    Returns points [x, y] which can be used to plot a circle
    of radius r, centred at a point c = [cx, cy].
    
    Example usage - circle centred at (1, -1), radius 0.5:
    
    x, y = circle_coords([1, -1], 0.5)
    plt.plot(x, y)
    '''
    t = np.linspace(0, 2*np.pi, 200)
    x = r * np.cos(t) + c[0]
    y = r * np.sin(t) + c[1]
    return x, y

# Transceiver coordinates
coords = np.array([[326798, 673275],
                   [313628, 678976],
                   [325585, 673905]], dtype=float)

# Time delays
dt = np.array([1.8708e-05, 0.000104115, 2.41092e-05])

# Speed of light (m/s)
c = 299792458.

# Distances (divide by 2 to account for the round trip)
dist = c * dt / 2

# Plot the circles
fig, ax = plt.subplots(figsize=(10, 10))
labels = ['Crags', 'Bridge', 'Scott']
for i in range(3):
    x, y = circle_coords(coords[i, :], dist[i])
    ax.plot(x, y, label=labels[i])

ax.axis('equal')
ax.set(xlim=[3.2e5, 3.3e5], ylim=[6.6e5, 6.8e5])
ax.legend()
plt.show()
```

---

## Exercise 6

```python
# Plot the circles
fig, ax = plt.subplots(figsize=(10, 10))
labels = ['Crags', 'Bridge', 'Scott']
for i in range(3):
    x, y = circle_coords(coords[i, :], dist[i])
    ax.plot(x, y, label=labels[i])

ax.axis('equal')
ax.set(xlim=[3.2e5, 3.3e5], ylim=[6.6e5, 6.8e5])
ax.legend()


# Function whose root we need -- we only use the first 2 transceivers
def F(x):
    return (x[0] - coords[:2, 0])**2 + (x[1] - coords[:2, 1])**2 - dist[:2]**2

# Jacobian matrix
def Jac(x):
    J = np.zeros([2, 2])
    J[:, 0] = 2 * (x[0] - coords[:2, 0])
    J[:, 1] = 2 * (x[1] - coords[:2, 1])
    return J


# Tolerance: 1 metre
tol = 1

# Initial guess
x = np.array([3.2e5, 6.7e5])
ax.plot(x[0], x[1], 'r+', markersize=10)

# Newton method
while np.linalg.norm(F(x)) > tol:
    x -= np.linalg.solve(Jac(x), F(x))
    ax.plot(x[0], x[1], 'r+', markersize=10)

plt.show()

print(f'My position is {x[0]:.0f} (Easting), {x[1]:.0f} (Northing).')
```

---

## Exercise 7

```python
def Y(N):
    return N / (2 + N**2)

def Yp(N):
    return 1 / (2 + N**2) - 2*N**2 / (2 + N**2)**2

def Ypp(N):
    return -6*N / (2 + N**2)**2 + 8*N**3 / (2 + N**2)**3

# Plot the function
fig, ax = plt.subplots()
x = np.linspace(0, 15, 1000)
ax.plot(x, Yp(x))
ax.set(xlabel='N', ylabel='Y\'(N)')
plt.show()

# Newton's method
N = 1
tol = 1e-12

while abs(Yp(N)) >= tol:
    N -= Yp(N) / Ypp(N)

print(f'The optimal nitrogen level is {N:.5f}.')
```

---

## Exercise 8

```python
mu = 1.2

def G(theta):
    return mu * np.sin(theta) + np.cos(theta)

def Gp(theta):
    return mu * np.cos(theta) - np.sin(theta)

def F(theta):
    return mu * 10 * 9.81 / G(theta)

def Fp(theta):
    return -F(theta) * Gp(theta) / G(theta)

def Fpp(theta):
    return F(theta) * (1 + 2 * Gp(theta)**2 / G(theta)**2)

# Plot the function
fig, ax = plt.subplots()
x = np.linspace(0, np.pi / 2, 1000)
ax.plot(x, F(x))
ax.set(xlabel=r'$\theta$', ylabel=r'$F(\theta)$')
plt.show()

# Newton's method
theta = 1
tol = 1e-12

while abs(Fp(theta)) >= tol:
    theta -= Fp(theta) / Fpp(theta)

print(f'The optimal angle is {theta/np.pi:.2f}pi radians.')
```

---

## Exercise 9

```python
# Minimise the distance between (1, 1) and a point (x0, tan(x0))

def F(x):
    return (x - 1)**2 + (np.tan(x) - 1)**2

def Fp(x):
    return 2 * (x - 1) + 2 / np.cos(x)**2 * (np.tan(x) - 1)

def Fpp(x):
    return 2 + 2 / np.cos(x)**2 * (2 * np.tan(x) * (np.tan(x) - 1) + 1/np.cos(x)**2)

# Plot the function
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
x = np.linspace(-np.pi/2, np.pi/2, 1000)
ax[0].plot(x, F(x))
ax[0].set(xlabel=r'$x$', ylabel=r'$F(x)$', ylim=[-1, 10])

ax[1].plot(x, np.tan(x), 'k-', label=r'$\tan(x)$')
ax[1].plot(1, 1, 'rx', label='Point (1, 1)')
ax[1].set(xlabel=r'$x$', ylabel=r'$y$', ylim=[-1, 2])

# Newton's method
x = 1
tol = 1e-12

while abs(Fp(x)) >= tol:
    x -= Fp(x) / Fpp(x)

ax[1].plot(x, np.tan(x), 'gx', label='Closest point')
ax[1].legend()
plt.show()

print(f'The point has coordinates ({x:.3f}, {np.tan(x):.3f}).')
```
