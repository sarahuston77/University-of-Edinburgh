import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def create_hills(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Outputs r(x i.e. location range) which is the altitude (or relief), for the given parameters (params)."""

    # Separate parameters into respective vectors
    H, u, sig = params

    # Broadcast through x values applying r(x) formula, which includes summing each H,u,sig 
    # None and axis = 1 keep track of values and perform actions on a different row, then sum the correct row
    r = np.sum(H / (sig * np.sqrt(2 * np.pi)) * np.exp(-((x[:, None] - u) ** 2) / (2 * (sig ** 2))), axis = 1)
    
    return r


def plot_hills(x: np.ndarray, r: np.ndarray, snow_depth = None):
    """Create a plot of a given hill range, using the points given in x. 
    Returns Figure and Axes objects containing the plot."""

    # Create plot of proper size
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot hill with given x/r vals from create_hills with proper labels/title
    ax.plot(x, r, label = "Hill", c = "hotpink")
    ax.set_title("Similated Hill With and Without Snow")
    ax.set_xlabel("X")
    ax.set_ylabel("Altitude (Meters)")

    # Plot snow if there is any
    if snow_depth is not None:
        snow_hill = r + snow_depth
        ax.plot(x, snow_hill, label = "Snowy Hill", c = "darkred")
    
    # Display legend
    ax.legend()

    return fig, ax


def estimate_snowfall(snow_depth: np.ndarray, params: np.ndarray, L: float, method: str):
    """Returns V(h) to approx exact vol of snowfall obtained using method given."""
    # Obtain the height using the fact that snow_depth is equally spaced
    h = L / (len(snow_depth) - 1)

    # Set up x-vals through linspace which creates equally spaced vals (start, stop, # points)
    x_vals = np.linspace(0, L, len(snow_depth))

    # Create hills underneath S so that we are able to effectively subtract
    r = create_hills(x_vals, params)
    S = snow_depth + r

    if method == 'riemann_left':
        # Multiplies by every value excluding the last times the width (h) and sums
        integral_S = np.sum(S[:-1]) * h
    else:
        # Trapezoid rule: width * (sum of all points excluding endpoints + 1/2 endpoints bc they are excluded in the summation loop)
        integral_S = h * (0.5 * S[0] + np.sum(S[1:-1]) + 0.5 * S[-1])

    # Use helper fn to get exact integral of r
    integral_rx = integral_r(params, x_vals)

    # Calculate V from given formula
    V = integral_S - integral_rx
    return V


def integral_r(params: np.ndarray, x_vals: np.ndarray) -> float:
    """Returns exact integral of r(x) and used as a helper fn to approx V(h)."""

    # Separate parameters into respective vectors
    H, u, sig = params

    # Broadcast through each point by None (helps to reshape and
    # perform operations on seperate row, then axis = 1 implies correct row summed) 
    # starting at one ahead and one less to stimulate a,b from x_vals.
    # Use erf fn to calculate exact integral of r. 
    r = np.sum(H * 0.5 * (erf((x_vals[1:, None] - u) / (sig * np.sqrt(2))) - erf((x_vals[:-1, None] - u) / (sig * np.sqrt(2)))), axis = 1)
    
    return np.sum(r)


def spaced_estimates(snow_depth: np.ndarray, params: np.ndarray, L: float) -> np.ndarray:
    """Uses both quadrature rules to estimate snow depth using only every 2^kth measurment available."""

    # Create a matrix for sol of two types of quadrature rules and 6 k val
    V2kh = np.ones((2, 7))

    # Loops through k vals 0..6
    for k in range(7):
        # Creates matrix (start, stop, step) for a given k
        idx = np.arange(0, len(snow_depth), 2 ** k)

        # Use indexs to gather correct points from snowdepth
        spaced_snow_depth = snow_depth[idx] 

        # Use estimate_snowfall fn to estimate vol for both methods and append matrix properly
        V2kh[0, k] = estimate_snowfall(spaced_snow_depth, params, L, 'riemann_left')
        V2kh[1, k] = estimate_snowfall(spaced_snow_depth, params, L, 'trapezoid')
    
    return V2kh


def min_points(params: np.ndarray, L: float, eps: float) -> int:
    """Returns the number of points needed to estimate a V using composite trapezoid rule with a given tolerance."""

    # Can only take the second derivative of a fn with at least three points
    n_min = 3

    # Starting at a baseline that will always begin while loop
    error = eps

    # Brute forces using diff sizes of lin spaces until the error is low enough
    while error >= eps:
        # X values with given # of points starting lowest
        x_vals = np.linspace(0, L, n_min)
        # Y values of a given relief
        r_points = create_hills(x_vals, params)
        # Space btwn points
        h = x_vals[1] - x_vals[0] 
        # Creates an array of all second derivatives
        second_derivatives = (r_points[2:] - 2*r_points[1:-1] + r_points[:-2]) / h**2 
        # Finds the largest second derviative because this is the lower bound of error 
        max_curvature = np.max(np.abs(second_derivatives)) 
        # Calculate error using given fn
        error = (h ** 2) * max_curvature * (L / 12)
        # Will break the loop before incrementing n_min again
        if error <= eps: break
        # Increase number of points since error was still higher than the boundary
        n_min += 1 

    return n_min