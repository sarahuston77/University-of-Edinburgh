import numpy as np
import matplotlib.pyplot as plt
import random as r


def monte_carlo(f, a: int|float, b: int|float, N: int) -> float:
    """Takes N random values to approx F() for each N and uses this to integrate."""
    
    random_points: np.ndarray = np.random.uniform(a, b, size=N)
    return np.sum(f(random_points) * ((b - a)/ N))


# Define general inputs and fn
a, b = -0.7, 0.9
M = 50


def f(x: int|float|np.ndarray) -> int|float|np.ndarray:
    return (np.exp(x - 1) * np.sin((2 * x) - 1.2)) + 1.5


def monte_carlo2d(f, a, b, N: int) -> float:
    """Adapts monte_carlo fn to estimate the integral of a function w/ two inputs."""

    x: np.ndarray = np.random.uniform(a[0], b[0], size = N)
    y: np.ndarray = np.random.uniform(a[1], b[1], size = N)

    return (b[0] - a[0]) * (b[1] - a[1]) * np.mean(f(x, y))

# Test the function
def f(x, y):
    return x * y

a, b = [0, 0], [1, 1]
I_exact = 1/4
I_approx = monte_carlo2d(f, a, b, 1000)
print(I_exact, I_approx)


def volume_ball(n, N):
    '''
    Estimates the volume of the n-ball using random uniform sampling.
    '''
    # Generate random points in the hypercube (each row is one point)
    x = 2 * np.random.random((N, n)) - 1

    # How many points are inside the ball
    # Points for which x_0^2 + x_1^2 + ... + x_n-1^2 <= 1
    inside = x[np.sum(x**2, axis=1) <= 1]

    # Compute the probability as the ratio inside/all
    p = inside.shape[0] / x.shape[0]

    # Volume is p * volume of the hypercube
    return p * 2**n
    