import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator


def k_solve(b: float = 1.0, d: float = 1.0, eps: float = 1.0):
    """Returns the value k within (0,1) such that rk(d) = b within tolerance eps."""
    # b = radius of right ring, d = distance btwn two rings, eps = tolerance

    # Create values of k to test, utlizing tolerance (range is (0, 1) non inclusive)
    k_tests = np.linspace(0 + eps, 1 - eps, int(1 // eps))

    # Broadcast through function recording all values of rk(k)
    b_guesses = np.cosh(d / k_tests) - np.sqrt(1 - k_tests ** 2) * np.sinh(d / k_tests)

    # Find the indices where the guess is within the tolerance of b
    correct_idx = np.abs(b_guesses - b) <= eps

    # Recored the correct k values
    k = k_tests[correct_idx]

    # If no correct k's, return None
    if k.size == 0:
        return None
    
    # If correct k's exist, calculate approx area for each
    approx_area = b * np.sinh(d / k) - (k ** 2 / 4) * np.sinh(2 * d / k) + (k * d / 2)
    
    # Use idx and min to choose the lowest approx area k
    return k[np.argmin(approx_area)]


def radius(s: float|np.ndarray, b: float = 1.0, d: float = 1.0) -> float:
    """Computes r(s) after finding correct k val such that r(d) = b."""
    # s = numbers btwn (0, d), b = radius of right ring, d = distance btwn two rings
    
    # Specified tolerance
    eps = 1e-7

    # Solve for correct k val
    k = k_solve(b, d, eps)

    # Calculate rs using k found previously
    rs = np.cosh(s / k) - np.sqrt(1 - k ** 2) * np.sinh(s / k)

    return rs


def surface(b: float = 1.0, d: float = 1.0, elevation = 30, figsize = 5, alpha = 0.05, n = 1000):
    '''
    Plots a 3D surface of the soap film.
    '''
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(figsize, figsize))

    # MAKE YOUR DATA HERE:
    # Create n equally spaced points for each range
    s = np.linspace(0, d, n)
    t = np.linspace(0, 2*np.pi, n)

    # Make a n, n shaped grid using s and t
    S, T = np.meshgrid(s, t)
    
    # Plug into radius fn and given equations to parameterize
    X = radius(S, b, d) * np.cos(T)
    Y = radius(S, b, d) * np.sin(T)
    Z = S

    # END MAKING YOUR DATA HERE.

    # Plot the surface
    ax.plot_surface(X, Y, Z,
                    cmap=cm.coolwarm,
                    antialiased=False,
                    alpha=alpha,
                    rstride=1,
                    cstride=n)
    
    # Customise the plot appearance
    ax.view_init(elev=elevation)
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()

    # Plot the rings
    ax.plot(X[:, 0], Y[:, 0], Z[:, 0], color='black', linewidth=4)
    ax.plot(X[:, -1], Y[:, -1], Z[:, -1], color='black', linewidth=4)
     
    return fig, ax


def bisection_method(func, a, b, tol):
    """
    Find the root of a function within an interval [a, b] using the bisection method.

    Parameters:
    - func: The function for which we are trying to find a root.
    - a: The start of the interval.
    - b: The end of the interval.
    - tol: The tolerance; the process is stopped when |b - a| < tol.

    Returns:
    - The root of the function (or the midpoint of the final interval).
    """
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise ValueError("The function must have different signs at the endpoints a and b.")

    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        fc = func(c)
        if fc == 0:  # c is a solution, done
            return c
        elif fc * fa < 0:  # Root lies between a and c
            b = c
            fb = fc
        else:  # Root lies between c and b
            a = c
            fa = fc

    return (a + b) / 2.0


def critical_d(b: float|np.ndarray, eps: float):
    """Returns the value of d where the stable soap film disintegrates."""
    from scipy.optimize import fsolve
    
    def equation(z, b):
        # Separate into pieces to solve
        k = np.sqrt(1 - ((np.cosh(z) - b) / np.sinh(z)) ** 2)
        k_sq = (np.cosh(z) - b)/np.sinh(z)

        # Calculate each part of the equation
        term1 = (-z * k * np.sinh(z) + z * k * np.cosh(z) * k_sq) / (k ** 2)
        term2 = (k * np.sinh(z)) / k_sq

        # Combine the terms to form the final equation
        equation_val = term1 + term2 
        return equation_val

    # Initial guess 
    initial_guess = b * 2
    
    # Use fsolve to find the root numerically based on type of input b
    if type(b) != int and type(b) != float:
        # Store z values
        z = np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            z[i], = fsolve(equation, initial_guess[i], args=(b[i]), xtol = eps)
    else:
        z, = fsolve(equation, initial_guess, args=(b), xtol = eps)
    
    # Solve for k and d
    k = np.sqrt((1 - ((np.cosh(z) - b)/np.sinh(z))**2))
    d = z * k
    return d



    # x0, x1 = 0, b * 2
    # z = (x0 + x1) / 2

    # while abs(equation(z, b)).all() > eps:
    # # If the midpoint and starting point are different signs, we want to move closer to a by making b the midpoint
    #     if equation(x0, b) * equation(z, b) < 0:
    #         x1 = z
    #     # Otherwise, we want to shift towards b by making a the midpoint
    #     else:
    #         x0 = z
    #     # Create new midpoint
    #     z = (x0 + x1) / 2