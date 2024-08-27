# Week 2 solutions

## Exercise 1

```python
# Create the 2x2 identity matrix
my_mat = [[1, 0], [0, 1]]

# Append one element to the 2 existing rows
my_mat[0].append(0)
my_mat[1].append(0)

# Append the 3rd row to the list of rows
my_mat.append([0, 0, 1])
```

---

## Exercise 2

```python
cat_names = ['Olive', 'Muffin', 'Butterscotch', 'Mister Beans', 'Nala']
cat_ages = [7, 3, 12, 11, 5]

# Find the position and display the name of the oldest cat
oldest_pos = cat_ages.index(max(cat_ages))
oldest_cat = cat_names[oldest_pos]
oldest_age = cat_ages[oldest_pos]
print(f'{oldest_cat} is the oldest cat ({oldest_age} years old).')

# Find the position and display the name of the youngest cat
youngest_pos = cat_ages.index(min(cat_ages))
youngest_cat = cat_names[youngest_pos]
youngest_age = cat_ages[youngest_pos]
print(f'{youngest_cat} is the youngest cat ({youngest_age} years old).')

# Find the average age of these cats
average_age = sum(cat_ages) / len(cat_ages)
print(f'The average age of these {len(cat_ages)} cats is {average_age} years old.')
```

---

## Exercise 3

```python
m = ['a', 'b', 'c', 'd', 'e']
m_back = m[::-1]
print(m_back)
```

---

## Exercise 4

```python
# We can get pi from the math module
import math

# Create a list of n-1 ones
n = 6
my_list = [1] * (n-1)

# Append pi, check length
my_list.append(math.pi)
print('Does the list have length n now?', len(my_list) == n)

# Change the value of the 3rd element
my_list[2] = sum(my_list[4:])
print(my_list)
```

---

## Exercise 5

The line `These are all the elements of a.` is now printed multiple times, because the `print()` command is now **part of the loop**. It is executed at every iteration of the loop, instead of just once after the loop is finished. This is because in Python, the level of indentation is what determines which lines are part of a loop, and where a loop finishes.

---

## Exercise 6

```python
# Set n and initialise P
n = 20
P = 1

# Loop from j=2 to j=n
for j in range(2, n+1):
    # Multiply each term in succession
    P *= j**3 + 5*j**2 - 3

# Display the result
print(P)
```

---

## Exercise 7

```python
import math

e = 1
n = 0

# Loop as long as e is not close enough to the exact value
while not math.isclose(math.exp(1), e, rel_tol=1e-6):
    n += 1                      # increment n by 1
    e += 1 / math.factorial(n)  # add the nth term of the series
    
    print(f'n = {n}')
    print(f'Exact value of exp(1): {math.exp(1):.6f}')
    print(f'Approximate value: {e:.6f}\n')

print(f'{n} iterations are needed.')
```

---

## Exercise 8

```python
def compute_P(n):
    '''
    Computes the product P for a value of n.
    '''
    # Initialise P
    P = 1

    # Loop from j=2 to j=n
    for j in range(2, n+1):
        # Multiply each term in succession
        P *= j**3 + 5*j**2 - 3

    # Return the result
    return P

# Test the function
print(compute_P(7))
```
