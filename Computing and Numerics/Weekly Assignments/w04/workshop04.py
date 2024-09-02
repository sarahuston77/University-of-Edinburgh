"""CNU Workshop Week 04."""

import numpy as np

def det_2(A: np.array) -> int:
    """Computes the determinant of a 2 X 2 matrix."""
    return (A[0, 0] * A[1, 1]) - (A[0, 1] * A[1, 0])


def minor(A: np.array, i: int, j: int) -> np.array:
    """Makes a copy matrix without row i and col j."""
    
    if A.shape[0] <= i or A.shape[1] <= j:
        raise ValueError(f'Cannot remove row {i}, column {j} as A has {A.shape[0]} rows and {A.shape[1]} columns.')
    
    Cij = np.delete(A, i, 0)
    Cij = np.delete(Cij, j, 1)
    return Cij


def det_ce(A: np.array) -> int:
    """Uses det_2(), minor(), and itself to compute det."""
    if A.shape == (2,2):
        return det_2(A)
    else:
        det = 0
        for j in range(A.shape[0]):
            det += det_ce(minor(A, 0, j)) * ((-1) ** (j + 1)) * A[0, j]
        return det


import time

A = np.random.random([10, 10])

t0: float = time.time()
det_ce(A)
t1: float = time.time() - t0
print(f"{t1:.6f}")

t2: float = time.time()
np.linalg.det(A)
t3: float = time.time() - t2
print(f"{t3:.6f}")