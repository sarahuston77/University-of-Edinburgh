import numpy as np

def polyfit_coeffs(x, y):
    '''Returns the coefficients a_i.'''

    # Create matrix size 4x4 to hold coef's from plugging into four equations i.e. (a0..a3 * coef)
    coef_matrix = np.zeros((len(x), len(x)))
    
    # 
    for idx in range(len(x)):
        coef_matrix[idx, 0] = (1 - (x[idx] ** 3))
        coef_matrix[idx, 1] = (x[idx] +(x[idx] ** 2))
        coef_matrix[idx, 2] = ((x[idx] ** 2) - x[idx])
        coef_matrix[idx, 3] = (1 + (x[idx] ** 3))

    # Solve (coef matrix) * (a0..a3) = (y0..y3) akin to Ax = b
    return np.linalg.solve(coef_matrix, y)