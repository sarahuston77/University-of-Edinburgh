"""Practice for week 04."""

def chess_board() -> None:
    """Inputs a square and returns the color."""

    square: str = input("Please input the location of the square: ")

    if ord(square[0]) < 97 or ord(square[0]) > 104 or int(square[1:]) > 8 or int(square[1:]) < 1:
        print("This square is not on the board.")
    elif (ord(square[0]) % 2 == 0 and int(square[1:]) % 2 == 0) or (ord(square[0]) % 2 == 1 and int(square[1:]) % 2 == 1):
        print(f'The square {square} is black.')
    else:
        print(f'The square {square} is white.')

# chess_board()

def joini(x: list) -> int:
    """Takes list of ints and merges into one int."""
    join_str: str = ""
    
    for num in x:
        join_str += str(num)
    return join_str

print(joini([2,34,56,7]))