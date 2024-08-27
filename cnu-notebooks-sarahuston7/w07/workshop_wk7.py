"""Workshop Week 7."""
import numpy as np

time = 52 * 5
time_step = 1 / 7
k = 0.1
K = 1000
m = 10
population = np.array(([300, 600, 900, 1200, 1500]))

def dp(p):
    deriv = k * p[:] * (1 - p[:] / K) - m
    for elem in deriv:
        if elem <= 0:
            idx = np.where(deriv == elem)
            break
    return deriv[:int(idx[0])]

print(dp(population))