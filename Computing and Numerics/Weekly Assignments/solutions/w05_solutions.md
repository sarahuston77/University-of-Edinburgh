# Week 5 tutorial solutions

---

## Exercise 1

```python
# Create an x-axis with 1000 points
x = np.linspace(-np.pi, np.pi, 1000)

# Evaluate the functions at all these points
f1 = np.sin(x)
f2 = np.tan(0.49 * x)
f3 = np.sin(x) * np.cos(2*x)

# Create the plots in the same axes
plt.plot(x, f1, 'r-.')
plt.plot(x, f2, 'g:')
plt.plot(x, f3, 'b--')

# Display the plot
plt.show()
```

---

## Exercise 2

```python
import matplotlib.pyplot as plt
import numpy as np

# Create an x-axis with 1000 points
x = np.linspace(-np.pi, np.pi, 1000)

# Evaluate the functions at all these points
f1 = np.sin(x)
f2 = np.tan(0.49 * x)
f3 = np.sin(x) * np.cos(2*x)

# Create a figure with 3 subplots
fig, ax = plt.subplots(1, 3, figsize=(10, 4))

# Plot each function in a different subplot
ax[0].plot(x, f1, 'r-.')
ax[1].plot(x, f2, 'g:')
ax[2].plot(x, f3, 'b--')

# Display the plot
plt.show()
```

---

## Exercise 3

```python
import matplotlib.pyplot as plt
import numpy as np

# Create an x-axis with 1000 points
x = np.linspace(-np.pi, np.pi, 1000)

# Evaluate the functions at all these points
f1 = np.sin(x)
f2 = np.tan(0.49 * x)
f3 = np.sin(x) * np.cos(2*x)

# Create a figure with 3 subplots
fig, ax = plt.subplots(1, 3, figsize=(10, 4))

# Plot each function in a different subplot
ax[0].plot(x, f1, 'r-.')
ax[1].plot(x, f2, 'g:')
ax[2].plot(x, f3, 'b--')

# Store y-axis label for each plot
y_labels = [r'$f_1(x)$', r'$f_2(x)$', r'$f_3(x)$']

# Set all 3 properties for the 3 plots
for i in range(3):
    ax[i].set_xlim([-np.pi, np.pi])
    ax[i].set_xlabel(r'$x$', fontsize=14)
    ax[i].set_ylabel(y_labels[i], fontsize=14)

# Make some space
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Display the plot
plt.show()
```

---

## Exercise 4

```python
import matplotlib.pyplot as plt
import numpy as np
import math

# Define a function for the truncated Maclaurin series
def trunc_cos(x, n):
    '''
    Return the truncated Maclaurin series for
    cos(x), with terms up until order n.
    '''
    cos_series = 0
    for k in range(n//2 + 1):
        # Add each term of the series up to nth order
        cos_series += (-1)**k * x**(2*k) / math.factorial(2*k)
    
    return cos_series


# Create an x-axis with 1000 points
x = np.linspace(-np.pi, np.pi, 1000)

# Create a figure
fig, ax = plt.subplots()

# Plot the requested functions
ax.plot(x, np.cos(x), 'k-', label=r'$\cos(x)$')
ax.plot(x, trunc_cos(x, 2), 'r--', label=r'Order 2')
ax.plot(x, trunc_cos(x, 4), 'g-.', label=r'Order 4')
ax.plot(x, trunc_cos(x, 6), 'b:', label=r'Order 6')

# Set axis properties
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.5, 1.5])
ax.set_xlabel(r'$x$', fontsize=12)
ax.legend()

plt.show()
```

---

## Exercise 5

```python
import matplotlib.pyplot as plt
import numpy as np

# Let's write a convenience function
def f(x):
    # Set coefficients
    a, b, c = -1, 3, 5
    
    # Compute the roots
    sqrt_delta = np.sqrt(b**2 - 4*a*c)
    roots = [(-b - sqrt_delta)/(2 * a), (-b + sqrt_delta)/(2 * a)]
    
    # Return the polynomial and the 2 roots
    return a*x**2 + b*x + c, roots

# Create an x-axis, compute f(x) and both roots
x = np.linspace(-4, 5, 100)
y, roots = f(x)

# Create the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

# Plot the function and the roots
ax.plot(x, y, '--', color='deepskyblue', label=r'$f(x) = -x^2 + 3x + 5$')
ax.plot(x, np.zeros(x.shape[0]), 'k-', label=r'$y = 0$')
ax.plot(roots[0], 0, 'magenta', label='First root', marker='^', markersize=10)
ax.plot(roots[1], 0, 'magenta', label='Second root', marker='^', markersize=10)

# Tidy up the plot
ax.set_xlim([-4, 5])
ax.set_ylim([y[0], 10])
ax.set_xticks(range(-4, 6))
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$f(x)$', fontsize=14)
ax.set_title('Polynomial roots', fontsize=14)
ax.legend(loc='lower center')
ax.grid(True)

plt.show()
```

---

## Exercise 6

Increasing the number of partitions $M$, i.e. decreasing the width of the partitions, is a straightforward way to obtain better accuracy using these methods. For instance, with $M = 300$, we get the correct value within $10^{-2}$.

We can calculate and plot the error on a log-log scale for different values of M:

```python
# You can put this function in integration.py,
# just be careful with the intg. prefix.
def test_accuracy(rule, f, a, b, I_exact):
    '''
    Estimate accuracy of a given rule (Riemann, midpoint, trapezoid),
    using a test function f over an interval [a, b].
    The exact integral is given by I_exact.
    '''
    # Test accuracy with different values of M: 4, 8, 16, 32...
    err = []
    M_vals = []
    for i in range(2, 11):
        M = 2**i
        M_vals.append(M)
        err.append(abs(I_exact - intg.estimate_integral(rule, f, a, b, M)))

    # Plot log(M) vs. log(err)
    fig, ax = plt.subplots()
    ax.plot(np.log(M_vals), np.log(err), 'gx', label='log(error)')
    ax.set(xlabel=r'$\log(M)$', ylabel=r'$\log(err)$', title=f'Method: {rule}')


    # Fit a line (a deg. 1 polynomial) through the points
    line_coeffs = np.polyfit(np.log(M_vals), np.log(err), 1)

    # Plot the line on the same graph
    x_plot = np.linspace(1, 8, 100)
    line_plot = np.polyval(line_coeffs, x_plot)
    ax.plot(x_plot, line_plot, 'r-', label='Line of best fit')

    ax.legend()

    print(f'The slope is {line_coeffs[0]:.6f}.')
    plt.show()

test_accuracy('riemann_L', f, a, b, I_exact)
test_accuracy('riemann_R', f, a, b, I_exact)
```
A line of slope $-1$ means that the error decreases **linearly** with increasing M. For instance, when M doubles, the error will decrease by half.

---

## Exercise 7

We can calculate and plot the error on a log-log scale for different values of M:

```python
test_accuracy('midpoint', f, a, b, I_exact)
```

A line of slope $-2$ means that the error decreases **quadratically** with increasing M. In other words, when M doubles, the error is divided by 4.

---

## Exercise 8

```python
test_accuracy('trap', f, a, b, I_exact)
```

A slope of -2 signifies that the error decreases with $M^2$. This means that the midpoint rule and the trapezoid rule both give a more accurate estimate than the left or right Riemann sums, for smaller values of $M$.
