def every_nth(n, p):
    """Creates list of nth integer's btwn p/2 to p, start p/2."""
    
    # list(range) takes (start, stop, step) to create a list of ints.
    return list(range(int(p / 2), p + 1, n))