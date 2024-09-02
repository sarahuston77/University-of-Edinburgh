import numpy as np

def iterate_method(F, x0: float, x1: float, kmax: int) -> float:
    """Root estimation method using recursion and starting from two initial guesses."""
    
    guesses = [x0, x1]

    idx = 1
    while idx < kmax + 1:
        if F(guesses[idx]) == F(guesses[idx - 1]):
            return guesses[idx]
        gx = guesses[idx] - ((F(guesses[idx]) * guesses[idx - 1]) / (F(guesses[idx]) - F(guesses[idx - 1])))
        guesses.append(gx)
        idx += 1

    return guesses[-1]


def trisection(F, a: float, b: float) -> float:
    """Uses a trisection algorithm to find an approxmation for a root using four iterations."""

    i = 0
    while i <= 4:
        d1 = a + (1 / 3)*(b - a)
        d2 = a + (2 / 3)*(b - a)
        if F(a)*F(d1) <= 0:
            b = d1
        elif F(d1)*F(d2) <=0:
            a = d1
            b = d2
        else:
            a = d2

    return (a + b) / 2
