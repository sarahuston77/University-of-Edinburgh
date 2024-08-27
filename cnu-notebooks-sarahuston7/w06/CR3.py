import numpy as np

def midpoint(t, a, b, M):
    '''Returns the integral of a generic function f calculated by the midpoint rule
       over the interval [a,b] using M subintervals as instructed.'''

    # Calculate space btwn the pairs of points (step)
    step = (a + b) / M

    # Use 1/2 step to shift start/end vals and create equally spaced array w/ M intervals for the x's of the midpoints 
    x_midpoints = np.linspace(a + (step / 2), b - (step / 2), M)

    # Plug in x-vals of midpoints into fn (height), multiply by the step (width), i.e. w * h = Area, sum each interval
    return sum(t(x_midpoints) * step)