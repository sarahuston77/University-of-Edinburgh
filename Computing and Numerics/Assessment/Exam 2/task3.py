import numpy as np
from scipy.special import bernoulli


def gamma_error_1(n_max: int) -> np.ndarray:
    """Returns En = Hn - log(n) - Euler-gamma for 1 to n_max."""
    
    # Hard code gamma to the degree of precision given
    gamma = 0.5772156649015329

    # Array of n_vals used to create the harmonic sum and broadcast through np.log
    n_vals = np.arange(1, n_max + 1, 1)

    # Cumsum fn adds each reciprical and moves through n_vals
    Hn = np.cumsum(1 / n_vals)

    # Given fn for En
    En = (Hn[:] - gamma - np.log(n_vals[:]))

    return En


def gamma_error_2(n_max: int) -> np.ndarray:
    """Computes E2,n which is the next order gamma error fn for a given n_max."""

    # Hard code gamma to the degree of precision given
    gamma = 0.5772156649015329

    # Array of n_vals used to create the harmonic sum and broadcast through np.log
    n_vals = np.arange(1, n_max + 1, 1)

    # Cumsum fn adds each reciprical and moves through n_vals
    Hn = np.cumsum(1 / n_vals)

    # Given fn for En
    En = (Hn[:] - gamma - np.log(n_vals[:]) - (1 / (2 * n_vals[:])))

    return En


def gamma_error(k: int, n_max: int) -> np.ndarray:
    """Returns the Ek,n which is the gamma error for a given k and n_max."""
    
    # Hard code const to the degree of precision given
    gamma = 0.5772156649015329

    # Array of n_vals and k_vals used to create the harmonic sum and broadcast through np.log
    n_vals = np.arange(1, n_max + 1, 1)
    k_vals = np.arange(1, k + 1, 1)

    # Cumsum fn adds each reciprical and moves through n_vals
    Hn = np.cumsum(1 / n_vals)

    # Collect bernouli sum values
    B = []

    # Go through n values bc cannot broadcast into diff shape, then n can acess each k_val using broadcasting
    for n in range(1, n_max + 1):
        B.append(np.sum((bernoulli(k)[1:]) / (k_vals * (n ** k_vals))))

    # Given fn for En
    En = (Hn[:] - gamma - np.log(n_vals[:]) - B[:])

    return En