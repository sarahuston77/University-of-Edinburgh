import numpy as np

def polyfit_coeffs(x, y):
    # Rewrite as p(x) = c0 + c1 x + c2 x^2 + c3 x^3
    # Set up the linear system Ac = y
    A = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            A[i, j] = x[i] ** j

    # Solve the system
    c = np.linalg.solve(A, y)

    # Set up a system to get the coefficients a from c
    A = np.array([[1, 0, 0, 1],
                  [0, 1, -1, 0],
                  [0, 1, 1, 0],
                  [-1, 0, 0, 1]], dtype=float)

    # Solve the system
    return np.linalg.solve(A, c)
