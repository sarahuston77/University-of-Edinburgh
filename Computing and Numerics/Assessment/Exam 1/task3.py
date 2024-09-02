import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def resting_state(u_init: float, v_init: float, eps: float, gamma: float, beta: float, I: float) -> tuple[(float, float)]:
    """Returns the values (u*, v*) for the cell in the resting cell for a given set of params."""

    # Define the general function
    def F(u, v):
        return np.array([(1 / eps) * (u - (u**3)/3 - v + I), eps * (u - gamma * v + beta)])

    # Create Jacobian matrix for given functions and inputs
    def Jac(u, v):
        J = np.zeros([2, 2])
        J[0, 0] = (1 / eps) * (1 - (2 / 3) * u**2)
        J[0, 1] = (1 / eps) * (-1)
        J[1, 0] = eps
        J[1, 1] = eps * - gamma
        return J
    
    # Initialise an array to store all the roots
    roots = []

    # Tolerance
    tol = 1e-8

    # Initial guesses
    uv = np.array([u_init, v_init])

    # Newton's method
    while np.linalg.norm(F(uv[0], uv[1])) >= tol:
        # Newton iteration
        e = -np.linalg.solve(Jac(uv[0], uv[1]), F(uv[0], uv[1]))
        uv += e
        
    # Store the results
    roots.append(uv)
    
    return uv

def solve_ODE(u0: float, v0: float, nmax: float, dt: float, eps: float, gamma: float, beta: float, I: float) -> tuple[(float, float)]:
    """Computes numerical solution (un, vn) ~ to sol for FitzHugh-Nagumo model using Forward Euler method."""
    
    # Store solution in array of specified size
    uv = np.zeros((2, nmax + 1))

    # Set initial conditions
    uv[0] = u0
    uv[1] = v0
    
    # Forward Euler method
    for n in range(nmax):
        uv[0, n + 1] = uv[0, n] + dt * ((1 / eps) * (uv[0, n] - (uv[0, n]**3)/3 - uv[1, n] + I))
        uv[1, n + 1] = uv[1, n] + dt * (eps * (uv[0, n] - gamma * uv[1, n] + beta))
    
    return uv


def plot_solutions(uv, dt):
    '''
    Plot solutions over time and in phase space.
    '''
    fig, ax = plt.subplots(1, 2, figsize=(8, 2))

    time = np.linspace(0, dt*uv.shape[1], uv.shape[1])

    # First plot with proper labels/legend for u and v
    ax[0].plot(time, uv[0], label = "$u_n$")
    ax[0].plot(time, uv[1], label = "$v_n$")
    ax[0].set_xlabel('Time(s)')
    ax[0].set_ylabel('Functions ($u_n$ & $v_n$)')
    ax[0].legend()

    # Second plot of u(t) vs v(t) with proper labels
    ax[1].plot(uv[0], uv[1])
    ax[1].set_xlabel('$u_n$')
    ax[1].set_ylabel('$v_n$')

    return fig, ax
