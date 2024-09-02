import numpy as np

# Replace the "pass" statements with your code for each function.
# Remember to write docstrings!

def square(n: int = 0) -> int:
    """Returns the int squared i.e. num of marbles in n x n square."""
    return n ** 2


def triangle(n: int = 0) -> int:
    """Returns the triangle number (int) i.e. num of marbles in n length equilateral triangle."""
    return n * (n + 1) // 2


def brute_force_search(m_max: int) -> list:
    """Takes int to create list w/ all pairs of S(m) = T(n) m <= m max."""

    # Later appended to store each pair S(m) = T(n)
    sol_pairs = []

    # Check m's less than m_max
    for m in range(m_max + 1):
        # Check n's within m <= n <= square rt * 2m bc this is a property of S(m) = T(n)
        for n in range(m, int(np.sqrt(2) * m) + 1):
            # Use triangle and square fns to compare S(m) == T(n) and add to list if True
            if triangle(n) == square(m):
                sol_pairs.append([m, n])
    return sol_pairs


def floor_root(n: int|np.ndarray):
    """Takes square root of int and round down to nearest integer."""
    # Uses n + 0.5 to account for python rounding errors
    return (np.floor(np.sqrt(n + 0.5))).astype(int)

def is_square(n: int|np.ndarray):
    """Will determine whether or not an int/array is a perfect square"""
    # Calls upon floor_root to take square rt and round down i.e. if
    # it is a square number, the square of this will be the same as n
    return (floor_root(n) ** 2) == n


def triangle_search(m_max: int):
    """Faster approach to create list w/ all pairs of S(m) = T(n) m <= m max (input)."""
    
    # Later appended to store each pair S(m) = T(n)
    sol_pairs = []

    # Check m vals up to largest input m value
    for m in range(m_max + 1):
        # Use property of m values to reduce the search for n values
        if is_square(1 + (8 * (m ** 2))):
            # Create n using given property
            n = floor_root(((1 + (8 * (m ** 2))) - 1)) // 2
            # Add pairs to our solution list to keep track
            sol_pairs.append([m, n])

    return sol_pairs


def matrix_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solves system XA = B w/ A & B (n x n) to give solution X (n x n)."""
    # Tranpose entire system so that np is able to solve normally
    return (np.linalg.solve(A.T, B.T)).T


def triquadrigon(k: int) -> tuple[int, int]:
    """Solves S(mk) = T(nk) for a given k using recursion formula."""

    # Set up base case and sol to X using earlier results
    m, n = 1, 1
    a, b, c, d = 3, 2, 4, 3

    # Move through all the potential values, updating m and n to build off of
    for val in range(1, k):
        m, n = ((a * m) + (b * n) + 1), ((c * m) + (d * n) + 1)
    return m, n