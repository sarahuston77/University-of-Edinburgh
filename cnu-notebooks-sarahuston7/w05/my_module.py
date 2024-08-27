# __name__ is a built-in name assigned automatically at runtime by Python.
# Run this as a Python script to see what __name__ is:
print(f'From inside the module: __name__ is {__name__}')

if __name__ == '__main__':
    print(f'This is only ran if __name__ is __main__')
