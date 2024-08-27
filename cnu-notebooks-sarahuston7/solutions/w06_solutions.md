# Week 6 solutions

---

## Exercise 1

```python
import numpy as np

def quadrature(f, xk, wk, a, b):
    '''
    Approximates the integral of f over [a, b],
    using the quadrature rule with weights wk
    and nodes xk.
    
    Input:
    f (function): function to integrate (as a Python function object)
    xk (Numpy array): vector containing all nodes
    wk (Numpy array): vector containing all weights
    a (float): left boundary of the interval
    b (float): right boundary of the interval
    
    Returns:
    I_approx (float): the approximate value of the integral
        of f over [a, b], using the quadrature rule.
    '''
    # Define the shifted and scaled nodes
    yk = (b - a)/2 * (xk + 1) + a
    
    # Compute the weighted sum
    I_approx = (b - a)/2 * np.sum(wk * f(yk))
    
    return I_approx


# Define the interval, nodes, and weights
a, b = 0.5, 1.2
xk = np.array([-1/3, 1/3])
wk = np.array([1., 1.])

# Display the result with 6 decimal digits
I_approx = quadrature(np.arctan, xk, wk, a, b)
print(f'The integral of arctan(x) over [{a}, {b}] is approximately {I_approx:.6f}.')
```


---

## Exercise 2

```python
# Plot the function
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_plot, f_plot, 'k-', linewidth=2.5, label=r'$\cos(5x)$')

# Display the exact value of the integral
print(f'Exact value: I = {I_exact:.6f}')

# Find interpolating polynomials for 3, 6, and 12 points
N_vals = [3, 6, 12]
colours = ['tab:orange', 'tab:green', 'tab:red']
markers = ['s', 'd', 'x']

for i, N in enumerate(N_vals):
    # Create N nodes, equally spaced in [-1, 1]
    xk = np.linspace(-1, 1, N)
    
    # Find the interpolating polynomial coefficients (degree N-1)
    p_coeffs = np.polyfit(xk, f(xk), N-1)
    
    # Integrate the polynomial
    p_int = np.polyint(p_coeffs)
    I_approx = np.polyval(p_int, 1) - np.polyval(p_int, -1)
    print(f'{N} nodes: {I_approx:.6f}')
    
    # PLOTTING
    # Evaluate the polynomial using these coefficients, to plot it smoothly
    p_plot = np.polyval(p_coeffs, x_plot)
    
    # Plot the points and the polynomial
    ax.plot(xk, f(xk), markers[i], color=colours[i], markersize=10)
    ax.plot(x_plot, p_plot, color=colours[i], linewidth=1.5, label=fr'$p_{{{N-1}}}(x)$')

# Label the plot
ax.set(title='Interpolating polynomials of different degrees', xlabel=r'$x$')
ax.set_ylim([-2, 2])

# Move the left and bottom spines to x = 0 and y = 0, respectively.
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
# Hide the top and right spines.
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend()
plt.show()
```


---

## Exercise 3

```python
# Define the function
def f(x):
    return np.exp(x - 1) * np.sin(2*x - 1.2) + 1.5

# Define the interval, nodes, and weights
a, b = -0.7, 0.9
xk = np.array([-1, 1])
wk = np.array([1., 1.])

# Compute the approximation
I_approx = quadrature(f, xk, wk, a, b)

# Display the result with 6 decimal digits
print(f'The integral of f(x) over [{a}, {b}] is approximately {I_approx:.6f}.')
```


---

## Exercise 4

We know that, by definition, the degree of precision is at least 1. We can test whether it integrates polynomials of degree 2 and above to investigate the degree of precision.

For $p(x) = x^2$, the exact integral is

$$
\int_{-1}^1 x^2 \ dx = \frac{1}{3} (1^3 - (-1)^3) = \frac{2}{3}.
$$

The approximation with the trapezoid rule is
$$
w_0 p(-1) + w_1 p(1) = (-1)^2 + 1^2 = 2 \neq \int_{-1}^1 x^2 \ dx.
$$

The trapezoid rule integrates polynomials of degree up to 1 exactly, but not of degree 2, therefore the degree of precision is 1.


---

## Exercise 5

Since we have 1 node $x_0 = 0$, the degree of precision is at least 0, so the midpoint rule integrates polynomials of degree 0 (constant functions) exactly. Therefore we must have

$$
\int_{-1}^1 1 \ dx = 1 - (-1) = 2 = w_0.
$$

The weight is $w_0 = 2$.

To investigate the degree of precision, we continue testing polynomials of degree 1 and above.

Degree 1: $p(x) = x$

$$
\int_{-1}^1 x \ dx = \frac{1}{2} (1^2 - (-1)^2) = 0,
$$
$$
w_0 p(0) = 2 \times 0 = 0 = \int_{-1}^1 x \ dx,
$$

therefore the midpoint rule also integrates polynomials of degree 1 exactly.

Degree 2: $p(x) = x^2$

$$
\int_{-1}^1 x^2 \ dx = \frac{2}{3},
$$
$$
w_0 p(0) = 2 \times 0^2 = 0 \neq \int_{-1}^1 x^2 \ dx,
$$

therefore the midpoint rule does not integrate polynomials of degree 2. The degree of precision is 1.


---

## Exercise 6

Since we have 3 nodes, the degree of precision is at least 2, so the midpoint rule integrates polynomials of degree up to 2 exactly. Therefore we must have

$$
\int_{-1}^1 1 \ dx = 2 = w_0 + w_1 + w_2,
$$
$$
\int_{-1}^1 x \ dx = 0 = -w_0 + 0 + w_2,
$$
$$
\int_{-1}^1 x^2 \ dx = \frac{2}{3} = w_0 + 0 + w_2.
$$

The second equation gives $w_0 = w_2$, the third then gives $w_0 = w_2 = \frac{1}{3}$. Substituting into the first equation, we get $w_1 = \frac{4}{3}$.

```python
# Define the function
def f(x):
    return np.exp(x - 1) * np.sin(2*x - 1.2) + 1.5

# Define the interval, nodes, and weights
a, b = -0.7, 0.9
xk = np.array([-1., 0., 1.])
wk = np.array([1/3, 4/3, 1/3])

# Compute the approximation
I_approx = quadrature(f, xk, wk, a, b)

# Display the result with 6 decimal digits
print(f'The integral of f(x) over [{a}, {b}] is approximately {I_approx:.6f}.')
```


---

## Exercise 7

The DOP is at least 2. We continue testing polynomials of degree 3 and above.

Degree 3: $p(x) = x^3$

$$
\int_{-1}^1 x^3 \ dx = \frac{1}{4} (1^4 - (-1)^4) = 0,
$$
$$
w_0 p(-1) + w_1 p(0) + w_2 p(1) = -\frac{1}{3} + 0 + \frac{1}{3} = 0 = \int_{-1}^1 x^3 \ dx,
$$

therefore Simpson's rule also integrates polynomials of degree 3 exactly.

Degree 4: $p(x) = x^4$

$$
\int_{-1}^1 x^4 \ dx = \frac{1}{5} (1^5 - (-1)^5) = \frac{2}{5},
$$
$$
w_0 p(-1) + w_1 p(0) + w_2 p(1) = \frac{1}{3} + 0 + \frac{1}{3} = 0 \neq \int_{-1}^1 x^4 \ dx,
$$

therefore Simpson's rule does not integrate polynomials of degree 4. The degree of precision is 3.

```python
# Define interval, nodes, and weights
a, b = -1, 1
xk = np.array([-1., 0., 1.])
wk = np.array([1/3, 4/3, 1/3])

# Calculate the integral of polynomials of degree up to 4
I_exact = [2, 0, 2/3, 0, 2/5]
for n in range(5):
    # Define a Python function for the polynomial
    def p(x):
        return x**n

    # We could use a "lambda function" to define a function like this in one line:
    # p = lambda x: x**n

    # Compute the integral and error
    I_approx = quadrature(p, xk, wk, a, b)
    err = abs(I_exact[n] - I_approx)
    print(f'Error for degree {n}: {err:.6f}')
```


---

## Exercise 8

```python
def composite_trapz(f, a, b, M):
    '''
    Returns the approximation of the integral of f
    over [a, b], using the composite trapezoid rule
    with M equal-width partitions.
    '''
    # Find each sub-interval
    bounds = np.linspace(a, b, M+1)
    
    # Define weights and nodes for trapezoid rule
    xk = np.array([-1., 1.])
    wk = np.array([1., 1.])
    
    # Loop to compute each small integral
    I_approx = 0
    for i in range(M):
        I_approx += quadrature(f, xk, wk, bounds[i], bounds[i+1])
    
    return I_approx
```


---

## Exercise 9

```python
# Define the function
def f(x):
    return x * np.exp(x)

# Define the interval
a, b = -1, 1

# Use a loop to compute the approximation for different M
M_vals = np.logspace(1, 6, 6, base=2, dtype=int)
I_approx = np.zeros(len(M_vals))

for i, M in enumerate(M_vals):
    I_approx[i] = composite_trapz(f, a, b, M)

# Calculate the error and interval size
I_exact = 2 / np.exp(1)
err = np.abs(I_exact - I_approx)
h = (b - a) / M_vals

# Plot the results
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(h, err, 'bx')
ax.set(title='Error in composite trapezoid rule',
       xlabel='Size of intervals',
       ylabel='Absolute error',
       xscale='log',
       yscale='log')

# This looks linear, fit a straight line and find the slope
line_coeffs = np.polyfit(np.log(h), np.log(err), 1)
print(f'The slope is {line_coeffs[0]:.6f}.')

plt.show()
```

The log-log graph has a slope of 2, which means that when the size of the intervals is doubled (i.e. $M$ is halved), the error increases by a factor of 4:

$$
\log(\text{err}) = 2 \log(h) + \beta = \log(h^2) + \beta = \log(\alpha h^2),
$$

where $\alpha = e^{\beta}$ is some constant. Then, taking the exponential on both sides,

$$
\text{err} = \alpha h^2.
$$

This is evidence that the rate of convergence for this method is $r=2$.


---

## Exercise 10

```python
# Define the function
def F(x):
    return (1 + np.cos(np.tan(x))) ** 2

# Create the plot
x = np.linspace(0.0, 0.45 * np.pi, 2000)

fig, ax = plt.subplots()
ax.plot(x, F(x), "k-")
ax.set_xlim([x[0], x[-1]])
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$F \left( x \right)$", fontsize=12)

plt.show()

# Compute the derivative approximation
dx = 0.01
F_derivative_approx = (F(x + dx) - F(x)) / dx

# Compare to the exact derivative
F_derivative = np.loadtxt("F_derivative.txt")
F_derivative_error = F_derivative_approx - F_derivative

# Plot the derivative, approximation, and error
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

ax[0].plot(x, F_derivative, "k-", label=r"$F'(x)$, exact")
ax[0].plot(x, F_derivative_approx, "r-", label=r"$F'(x)$, approximated")
ax[0].set_xlim([x[0], x[-1]])
ax[0].set_xlabel(r"$x$", fontsize=12)
ax[0].set_ylabel(r"$F' \left( x \right)$", fontsize=12)
ax[0].legend(loc="upper left", fontsize=12)

ax[1].plot(x, F_derivative_error, "k-", label="error")
ax[1].set_xlim([x[0], x[-1]])
ax[1].set_xlabel(r"$x$", fontsize=12)
ax[1].set_ylabel(r"$F' \left( x \right)$ error", fontsize=12)
ax[1].legend(loc="lower left", fontsize=12)

plt.show()
```

The error is largest at the right edge of the domain, where the second derivative of $F$ has large magnitude; we will see why next week.


---

## Exercise 11

```python
# Compute the derivative approximation
dx = 0.01
F_derivative_approx_centred = (F(x + dx) - F(x - dx)) / (2 * dx)

# Compare to the exact derivative
F_derivative_error_centred = F_derivative_approx_centred - F_derivative

# Plot the error for both methods
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, F_derivative_error, "r-", label="Forward difference")
ax.plot(x, F_derivative_error_centred, "g-", label="Centred difference")
ax.set_xlim([x[0], x[-1]])
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$F' \left( x \right)$ error", fontsize=12)
ax.legend(loc="lower left", fontsize=12)

plt.show()
```

The centred approximation also shows variations at the same places, but much lesser in magnitude. Both difference approximations estimate the derivative by computing the slope of a line drawn between 2 points which are near each other. It would make sense that this slope is more accurately estimated by drawing a line between 2 points on either side of the point of interest, instead of between the point of interest and the point to the right of it.
