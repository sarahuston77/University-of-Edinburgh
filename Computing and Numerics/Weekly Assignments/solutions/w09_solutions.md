# Week 9 tutorial solutions

---

## Exercise 1

```python
import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return -x**2 - 3*x + 2

# Calculate the roots
sqrt_delta = np.sqrt((-3)**2 + 4*2)
xL = -0.5 * (3 + sqrt_delta)
xR = -0.5 * (3 - sqrt_delta)

# Plot the function and the roots
xmin, xmax = -5, 2
x = np.linspace(xmin, xmax, 1000)

fig, ax = plt.subplots()
ax.axhline(0.0, color=[0.5, 0.5, 0.5])
ax.plot(x, F(x), 'k-')
ax.plot([xL, xR], [F(xL), F(xR)], 'ro')
ax.set(xlabel=r'$x$', ylabel=r'$F(x)$', xlim=[xmin, xmax])
plt.show()
```

---

## Exercise 2

```python
def G1(x):
    return 2 / (x + 3)

def G2(x):
    return 2 - 2*x - x**2

# Iterate with G1(x) and G2(x)
x1 = [-4.]
x2 = [-4.]
for k in range(8):
    # Compute the new guess from the previous guess, and append to the list
    x1.append(G1(x1[-1]))
    x2.append(G2(x2[-1]))

# Display the results
print(x1)
print(x2)
print(f'Exact value: {xR}')
```

Only the first method seems to converge to the positive root, even though we started with an initial guess closer to the negative root. The second method doesn't converge at all.


---

## Exercise 3

```python
def G1_prime(x):
    return (-2) / (x + 3)**2

def G2_prime(x):
    return -2*x - 2

xmin, xmax = -5, 5
x = np.linspace(xmin, xmax, 500)
fig, ax = plt.subplots(figsize=(12, 8))
ax.axhline(0.0, color=[0.5, 0.5, 0.5])
ax.axhline(1.0, color=[0.7, 0.8, 0.9], linestyle='--')
ax.plot(x, abs(G1_prime(x)), 'b-', label=r'$G_1\'(x)$')
ax.plot(x, abs(G2_prime(x)), 'g-', label=r'$G_2\'(x)$')
ax.plot([xL, xR], [F(xL), F(xR)], 'ro', label='Exact roots')
ax.set(xlabel=r'$x$', ylabel=r'$F(x)$', xlim=[xmin, xmax], ylim=[-0.5, 5])
ax.legend()
plt.show()
```

Using $G_1(x)$ with an initial guess $x_0 > -1.5$ approximately guarantees convergence towards the positive root. No initial guess could lead to convergence to the negative root, since there is no neighbourhood around $x_L$ where $|G_1(x)|<1$.

For $G_2(x)$, neither root has a neighbourhood where $|G_2(x)|<1$, therefore using $G_2(x)$ will not lead to convergence, no matter the initial guess.


---

## Exercise 4

```python
import numpy as np
import matplotlib.pyplot as plt


def F(x):
    return x - (x ** 2) * np.sin(x)


x_plot = np.linspace(-2.0, 2.0, 1000)
fig, ax = plt.subplots(figsize=(8, 5))

ax.axhline(0.0, color=[0.7, 0.7, 0.7])
ax.plot(x_plot, F(x_plot), "k-")
ax.set(xlabel=r"$x$", ylabel=r"$F \left( x \right)$", xlim=[-2, 2])
plt.show()
```

---

## Exercise 5

```python
def G(x, alpha):
    return (1.0 + alpha) * x - alpha * (x ** 2) * np.sin(x)


# Initialisation
x = 1.0
alpha = 1.2

while True:
    x_new = G(x, alpha)
    
    # Convergence achieved
    if abs(x_new - x) < 1.0e-12:
        break
        
    # Update value for next iteration
    x = x_new

x_star = x_new
print(f"Root = {x_star}")
```

---

## Exercise 6

```python
from scipy.optimize import fsolve

# Ground truth solution
x_star_fsolve = fsolve(F, 1.1, xtol=1.0e-14)

# Initialisation
x = 1.0
alpha = 1.2
error_norm = [abs(x - x_star_fsolve)]
its = 0

# Fixed point iteration
while True:
    its += 1
    x_new = G(x, alpha)
    error_norm.append(abs(x_new - x_star_fsolve))
    
    # Convergence achieved
    if abs(x_new - x) < 1.0e-12:
        break
        
    # Update value for next iteration
    x = x_new

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(its + 1), error_norm, "k+", label=r"Fixed-point, $\alpha = 1.2$")
ax.set(xlabel="Iteration", ylabel="Error magnitude")
ax.set_yscale("log")
ax.legend()
plt.show()
```


---

## Exercise 7

```python
# Initialise the bisection method
a = 0.5
b = 1.5
error_norm = [abs(0.5 * (a + b) - x_star_fsolve)]
its = 0

# Bisection method
while abs(a - b) >= 1.0e-12:
    its += 1
    c = 0.5 * (a + b)
    error_norm.append(abs(c - x_star_fsolve))

    if F(a) * F(c) <= 0.0:
        b = c
    else:
        a = c

# Plot on the same axes as above
ax.plot(range(its + 1), error_norm, "r+", label="Bisection")
ax.legend()
plt.show()
```

In this example bisection converges more rapidly than fixed-point iteration.


---

## Exercise 8

```python
# Ground truth solution
x_star_fsolve = fsolve(F, 1.1, xtol=1.0e-14)

# Initialisation
x = 1.0
alpha = 0.8
error_norm = [abs(x - x_star_fsolve)]
its = 0

# Fixed point iteration
while True:
    its += 1
    x_new = G(x, alpha)
    error_norm.append(abs(x_new - x_star_fsolve))
    
    # Convergence achieved
    if abs(x_new - x) < 1.0e-12:
        break
        
    # Update value for next iteration
    x = x_new

ax.plot(range(its + 1), error_norm, "b+", label=r"Fixed-point, $\alpha = 0.8$")
ax.legend()
plt.show()
```

With $\alpha = 0.8$ the convergence is now faster than with the bisection method. The choice of iteration function seems to strongly influence the rate of convergence.


---

## Exercise 9

```python
import matplotlib.pyplot as plt
import numpy as np


def F(x):
    return np.exp(0.1 * x) * np.sin(4.0 * np.pi * x) + x**2 + 0.5


x = np.linspace(0.0, 1.0, 1000)
fig, ax = plt.subplots()
ax.axhline(0.0, color=[0.7, 0.7, 0.7])
ax.plot(x, F(x), "r-", label=r'$F(x)$')
ax.set(xlim=[0.0, 1.0], xlabel=r"$x$", ylabel=r"$F \left( x \right)$")
plt.show()
```

$F(x)$ has two roots in this interval, at approximately $x = 0.3$ and $x = 0.45$. Both of these are simple roots.


---

## Exercise 10

```python
def Fp(x):
    # Note that the backslash "\" here allows code to be split over several lines
    return 0.1 * np.exp(0.1 * x) * np.sin(4.0 * np.pi * x) + \
        4.0 * np.pi * np.exp(0.1 * x) * np.cos(4.0 * np.pi * x) + 2.0 * x

def G(x):
    return x - F(x) / Fp(x)

# Draw G(x) - x on the same plot as above
ax.plot(x, G(x) - x, "g-", label=r'$G(x) - x$')
ax.set(ylim=[-1, 1])
ax.legend()
plt.show()
```

---

## Exercise 11

```python
# The roots are close to 0.3 and 0.45, respectively
for x0 in [0.3, 0.45]:
    
    # Initial guess
    x = x0
    
    # Loop until convergence
    while True:
        x_new = G(x)
        
        # Convergence achieved
        if abs(x_new - x) < 1.0e-14:
            break
        
        # Update for next iteration
        x = x_new
    
    # Display the value of the root
    print(f"Root = {x_new:.16e}")
```


---

## Exercise 12

```python
# Initialise a, the initial guess x0, and the number of iterations
a = 5.0
x = 1.0
its = 0

# Loop until convergence
while True:
    its += 1
    x_new = x - (np.exp(x) - a) / np.exp(x)
    
    # Convergence achieved
    if abs(x_new - x) < 1.0e-14:
        break
    
    # Update for next iteration
    x = x_new

print(f"log 2 is approximately {x_new:.16e}")
print(f"Number of iterations = {its}")
```


---

## Exercise 13

```python
# Initialise a, the initial guess x0, and the number of iterations
a = 5.0
x = 1.0
its = 0

# Start a list to store the error
err = [abs(x - np.log(a))]

# Loop until convergence
while True:
    its += 1
    x_new = x - (np.exp(x) - a) / np.exp(x)
    
    err.append(abs(x_new - np.log(a)))
    
    # Convergence achieved
    if abs(x_new - x) < 1.0e-14:
        break
    
    # Update for next iteration
    x = x_new

# Convert to Numpy array and remove last value (it's computed as zero)
err = np.array(err[:-1])
print(err)

# One way to find the order of convergence
fig, ax = plt.subplots()
ax.plot(np.log(err[:-1]), np.log(err[1:]), 'bx')
plt.show()

slope, _ = np.polyfit(np.log(err[:-1]), np.log(err[1:]), 1)
print(f'The order of convergence is p = {slope:.5f}')
```

---

## Exercise 14

```python
# Initialise a and the initial guesses x0
a = 5.0
x0 = range(-1, 8, 2)

# Prepare the plot
fig, ax = plt.subplots()

for x in x0:
    its = 0
    title = f'x0 = {x}'

    # Start a list to store the error
    err = [abs(x - np.log(a))]

    # Loop until convergence
    while True:
        its += 1
        x_new = x - (np.exp(x) - a) / np.exp(x)

        err.append(abs(x_new - np.log(a)))

        # Convergence achieved
        if abs(x_new - x) < 1.0e-14:
            break

        # Update for next iteration
        x = x_new
    
    # Plot the error
    ax.plot(range(its+1), err, linestyle='', marker='x', label=title)

ax.set(xlabel='Iterations', ylabel='Absolute error')
ax.legend()
plt.show()
```

For initial guesses further away from the root, quadratic convergence isn't observed immediately, but only after the guess is sufficiently close to the root.
