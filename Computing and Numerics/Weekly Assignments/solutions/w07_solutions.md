# Week 7 solutions

---

## Exercise 1

We take the Taylor expansion of $F(x - \Delta x)$ about $x$:

$$
  F \left( x - \Delta x \right) = F \left( x \right) - \Delta x F' \left( x \right) + \frac{(-\Delta x)^2}{2} F'' \left( x \right) + \frac{(-\Delta x)^3}{6} F'''\left(x\right) + \dots
$$

Since we are seeking an approximation to $F'(x)$, we can rearrange the above to make $F'(x)$ the subject:

\begin{align}
F'(x) &= \frac{1}{\Delta x}\left( F(x) - F(x - \Delta x) + \frac{\Delta x^2}{2}  F'' \left( x \right) - \frac{\Delta x^3}{6} F'''\left(x\right) + \dots \right) \\
&= \underbrace{\frac{F(x) - F(x - \Delta x)}{\Delta x}}_{:= D_{-1}(x)} + \frac{\Delta x}{2}  F'' \left( x \right) - \frac{\Delta x^2}{6} F'''\left(x\right) + \dots
\end{align}

The error is

$$
  D_{-1}(x) - F' \left( x \right)
  = -\frac{\Delta x}{2}  F'' \left( x \right) + \frac{\Delta x^2}{6} F'''\left(x\right) - \dots
  = O(\Delta x),
$$

therefore $D_{-1}(x)$ is also first order accurate.

---

## Exercise 2

We have

\begin{align}
  F \left( x + \Delta x \right) &= F \left( x \right) + \Delta x F' \left( x \right) + \frac{\Delta x^2}{2} F'' \left( x \right) + \frac{\Delta x^3}{6} F'''\left(x\right) + O(\Delta x^4) \\
  F \left( x - \Delta x \right) &= F \left( x \right) - \Delta x F' \left( x \right) + \frac{\Delta x^2}{2} F'' \left( x \right) - \frac{\Delta x^3}{6} F'''\left(x\right) + O(\Delta x^4)
\end{align}

Subtracting the second equation from the first, we obtain

$$
F(x + \Delta x) - F(x - \Delta x) = 2\Delta x F'(x) + \frac{\Delta x^3}{3} F'''(x) + O(\Delta x^4)
$$

Rearranging to get an expression for $F'(x)$,

$$
F'(x) = \frac{F(x + \Delta x) - F(x - \Delta x)}{2 \Delta x} + \frac{\Delta x^2}{6} F'''(x) + O(\Delta x^3)
= D_C(x) + O(\Delta x^2),
$$

therefore $D_C(x)$ is second order accurate.


---

## Exercise 3

To ensure that $D_1(x) - F'(x) = 0$ $\forall x$ and $\forall \Delta x \neq 0$, we must have

$$
 \frac{\Delta x}{2}  F'' \left( x \right) + \frac{\Delta x^2}{6} F'''\left(x\right) + \dots = 0.
$$

All derivatives of $F$ of order 2 or greater must be zero $\forall x$, therefore $F(x)$ must be a linear function $F(x) = \alpha x + \beta$, with $\alpha, \beta \in \mathbb{R}$.


---

## Exercise 4

We have
\begin{equation*}
  F' \left( x \right) = -2 x e^{-x^2},
\end{equation*}
which leads to
\begin{equation*}
  F' \left( -\tfrac{1}{2} \right) = e^{-\tfrac{1}{4}}.
\end{equation*}

```python
import numpy as np
import matplotlib.pyplot as plt

# Convenience functions for F(x) and F'(x)
def F(x):
    return np.exp(-x ** 2)

def F_derivative(x):
    return -2.0 * x * F(x)

# Test different values of the step size
x = 0.5
dx = np.array([0.04, 0.02, 0.01, 0.005])

# Calculate the FD approximation for all step sizes at once
F_derivative_approx = (F(x + dx) - F(x)) / dx

# Calculate the absolute error
F_derivative_error = np.abs(F_derivative_approx - F_derivative(x))

# Plot the results
fig, ax = plt.subplots()
ax.plot(dx, F_derivative_error, "kx")

# Label and tidy up the plot
ax.set(xlabel=r"$\Delta x$", ylabel=r"$F' \left( x \right)$ error magnitude", title="Forward difference")
ax.set_xlim([0.0, dx.max() * 1.1])
ax.set_ylim([0.0, 0.02])

plt.show()
```

The errors decrease approximately linearly with the step size $\Delta x$, falling close to a straight line in the plot, particularly for small $\Delta x$. In fact, we can look at the ratios between successive values of the error:

```python
print(F_derivative_error[1:] / F_derivative_error[:-1])
```

When the step size is halved, the error magnitude is also halved. This all suggests that the forward difference approximation is first order accurate.

Performing the same experiments with the centred difference approximation:

```python
# Calculate the FD approximation for all step sizes at once
F_derivative_approx = (F(x + dx) - F(x - dx)) / (2 * dx)

# Calculate the absolute error
F_derivative_error = np.abs(F_derivative_approx - F_derivative(x))

# Print the ratios between successive error values
print(F_derivative_error[1:] / F_derivative_error[:-1])

# Plot the results
fig, ax = plt.subplots()
ax.plot(dx, F_derivative_error, "kx")

# Label and tidy up the plot
ax.set(xlabel=r"$\Delta x$", ylabel=r"$F' \left( x \right)$ error magnitude", title="Centred difference")
ax.set_xlim([0.0, dx.max() * 1.1])
ax.set_ylim([0.0, 0.0012])

plt.show()
```

For the centred difference approximation, halving the step size approximately divides the error magnitude by four; the plot of error vs. step size is a parabola. This is expected, since $D_C(x)$ is second order accurate. Plotting the log error vs. log step size should reveal a line with slope 2:

```python
fig, ax = plt.subplots()
ax.plot(np.log(dx), np.log(F_derivative_error), "kx")

# Label the plot
ax.set(xlabel="$\log(\Delta x)$", ylabel="Log error magnitude", title="Centred difference")

# Compute and print the slope of the line
print(np.polyfit(np.log(dx), np.log(F_derivative_error), 1)[0])

plt.show()
```

---

## Exercise 5

Starting from
$$
F''(x) \approx \frac{F'(x + \Delta x) - F'(x)}{\Delta x},
$$

we now use $D_{-1}(x)$ to approximate $F'(x + \Delta x)$ and $F'(x)$, using the same step size:

$$
F''(x) \approx \frac{\frac{F(x + \Delta x) - F(x + \Delta x - \Delta x)}{\Delta x} - \frac{F(x) - F(x - \Delta x)}{\Delta x}}{\Delta x}
= \frac{F(x + \Delta x) - 2F(x) + F(x - \Delta x)}{\Delta x^2}
:= D^{(2)}_C(x).
$$

This final expression gives us a centred difference approximation of the second derivative $F''(x)$. To investigate its accuracy, we use the Taylor expansions of $F(x + \Delta x)$ and $F(x - \Delta x)$ about $F(x)$ (see solution to Exercise 2). Substituting these into $D^{(2)}_C(x)$ to calculate the error, we obtain

$$
D^{(2)}_C(x) - F''(x)
= \frac{1}{\Delta x^2} \left( \Delta x^2 F''(x) + O(\Delta x^4) \right)
= O(\Delta x^2).
$$

The terms in odd powers of $\Delta x$ cancel out, and this approximation is second order accurate.

---

## Exercise 6

```python
import numpy as np
import matplotlib.pyplot as plt

# Convenience functions for F(x) and F''(x)
def F(x):
    return np.exp(-x ** 2)

def F_second(x):
    return 2.0 * F(x) * (2.0 * x**2 - 1)

# Test different values of the step size
x = 0.5
dx = np.array([0.04, 0.02, 0.01, 0.005])

# Calculate the FD approximation for all step sizes at once
F_second_approx = (F(x + dx) - 2 * F(x) + F(x - dx)) / dx**2

# Calculate the absolute error
F_second_error = np.abs(F_second_approx - F_second(x))

# Plot the results in log-log scale
fig, ax = plt.subplots()
ax.plot(np.log(dx), np.log(F_second_error), "kx")

# Label the plot
ax.set(xlabel="$\log(\Delta x)$", ylabel="Log error magnitude", title="Centred difference (second derivative)")

# Compute and print the slope of the line
print(np.polyfit(np.log(dx), np.log(F_second_error), 1)[0])

plt.show()
```

The log of the error is approximately proportional to 2 times the log of the step size, which confirms that the approximation is second order accurate.


---

## Exercise 7

```python
import matplotlib.pyplot as plt
import numpy as np

# Set the required parameters
N = 100
dt = 0.1
gamma = 1.0

# Initialise an empty vector psi to store the time series
psi = np.zeros(N + 1)

# Initialise a time vector with N+1 time steps, and step size dt
t = np.linspace(0.0, N * dt, N + 1)

# Compute the solution iteratively
psi[0] = 1.0
for n in range(N):
    psi[n + 1] = (1.0 - gamma * dt) * psi[n]

# Plot the results
fig, ax = plt.subplots()
ax.plot(t, psi, "k-")
ax.set(xlabel=r"$t$", ylabel=r"$\psi_n$")

plt.show()
```


---

## Exercise 8

If $\gamma \Delta t > 1$ then the numerical approximation will under-shoot and oscillate about zero. If $\gamma \Delta t > 2$ then the under-shoot will amplify with each step, leading to a "blow-up". This is a type of numerical instability.


---

## Exercise 9

For instance, the backward difference approximation leads to

$$
  \psi_{n} = \frac{1}{1 + \gamma \Delta t} \psi_{n-1} \quad \textrm{ for } n = 1, 2, 3, \ldots
$$

The trapezoid rule approximation, seen at the end of Video 1, leads to

$$
  \psi_{n+1} = \frac{1 - \frac{\gamma \Delta t}{2}}{1 + \frac{\gamma \Delta t}{2}} \psi_{n} \quad \textrm{ for } n = 0, 1, 2, \ldots
$$

```python
import matplotlib.pyplot as plt
import numpy as np

# Set the required parameters
N = 100
dt = 0.1
gamma = 1.0

# Initialise an empty vector psi to store the time series
psi = np.zeros(N + 1)

# Initialise a time vector with N+1 time steps, and step size dt
t = np.linspace(0.0, N * dt, N + 1)

# Compute the solution iteratively
psi[0] = 1.0
for n in range(N):
    psi[n + 1] = (1.0 - gamma * dt) * psi[n]

# Backward Euler
psi_back = np.zeros(N + 1)
psi_back[0] = 1.0
for n in range(N):
    psi_back[n + 1] = 1 / (1.0 + gamma * dt) * psi_back[n]

# Trapezoid rule
psi_trapz = np.zeros(N + 1)
psi_trapz[0] = 1.0
for n in range(N):
    psi_trapz[n + 1] = (1.0 - 0.5 * gamma * dt) / (1.0 + 0.5 * gamma * dt) * psi_trapz[n]

# Exact solution
psi_exact = psi[0] * np.exp(-gamma * t)

# Plot all three together with the exact solution
%matplotlib notebook
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(t, psi_exact, 'k-', label='Exact solution')
ax[0].plot(t, psi, '--', label='Forward Euler')
ax[0].plot(t, psi_back, ':', label='Backward Euler')
ax[0].plot(t, psi_trapz, '-.', label='Trapezoid rule')
ax[0].set(xlabel=r"$t$", ylabel=r"$\psi_n$")
ax[0].legend()

ax[1].plot(t, np.abs(psi_exact - psi), '--', label='Forward Euler')
ax[1].plot(t, np.abs(psi_exact - psi_back), ':', label='Backward Euler')
ax[1].plot(t, np.abs(psi_exact - psi_trapz), '-.', label='Trapezoid rule')
ax[1].set(xlabel=r"$t$", ylabel="Error magnitude")
ax[1].legend()

plt.show()
```

At first glance, the trapezoid rule seems to give a much more accurate solution than the forward or backward Euler methods. Testing different values of $\gamma \Delta t$ reveals that the methods have different stability conditions; values which cause instability with the forward Euler method don't seem to cause an issue for backward Euler or the trapezoid rule.
