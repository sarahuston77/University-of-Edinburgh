# """First Qquiz practice."""

# def crop_list(m: list, start: int, stop: int) -> list and list:
#     """Will create a new list from start to stop. Then creates a list of remaining elements."""
#     m_middle: list = m[start:stop]
#     m_ends: list = []
#     for element in m:
#         if element not in m_middle:
#             m_ends.append(element)
#     return m_middle, m_ends


# m_middle, m_ends = crop_list([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 2, 5)
# print(m_middle)
# print(m_ends)


# import math


# def taylor_ln(N: int = 1, x: float = 1.0) -> None:
#     """Calculates value of PN(x) with error."""
#     value_list: list[float] = []
#     i: int = 1
#     while i < N:
#         value_list.append(((-1 ** (i + 1)) * ((x - 1) ** i)) / i)
#         i += 1
#     error: float = abs(round(math.log(x) - sum(value_list), 7))
#     print(f'For N = {N}, the Taylor polynomial equals {round(sum(value_list), 7)} with an error of {error}.')
#     return None


# def taylor_polynomial(x: float = 1.0) -> None:
#     """Uses a for loop to enter taylor_ln for 2-20 times."""
#     num_list: list[int] = range(0,20,2)
#     for N in num_list:
#         taylor_ln(N, x)
#     return None
    

# taylor_polynomial(1.5)

    
def sequence_element(n: int = 1) -> int:
    """Performs given integer sequence using n."""
    seq_list: list[int] = [1, 2]
    i: int = 2
    while i <= n:
        if (seq_list[i - 1] % 2) == 0:
            seq_list.append((3 * seq_list[i - 2]))
        else:
            seq_list.append((2 * seq_list[i - 1]) - seq_list[i - 2])
        i += 1
    return seq_list[n]

# print(sequence_element(1))
# print(sequence_element(2))
# print(sequence_element(3))
print(sequence_element(7))

