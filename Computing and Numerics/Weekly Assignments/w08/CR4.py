def polynomial_with_roots(N, m):
    """Returns the power series coefficient am as an int."""
    
    # Initialize beginning values of polynomial
    LHS = [1, 1]

    # Bottom up approach: (new left hand side) * (qx + 1)
    # New LHS is updated each iteration along with q
    for q in range(2, N + 1):
        # The first value doesn't follow the same pattern
        newLHS = [LHS[0] * q]
        for idxL in range(1, len(LHS)):
            newLHS.append(LHS[idxL - 1] + LHS[idxL] * q)
        # The last value is always in LHS is always 1
        newLHS.append(1)
        LHS = newLHS

    # The bottom up approach reverses the list i.e., must acess correct idx
    return LHS[len(newLHS) - m - 1]