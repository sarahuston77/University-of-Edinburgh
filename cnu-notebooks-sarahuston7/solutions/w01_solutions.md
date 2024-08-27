# Week 1 solutions

---

## Exercise 1

```python
print('I have solved Exercise 1!')
```


---

## Exercise 2

* `+`, `-` are addition and subtraction,
* `*`, `/` are multiplication and division,
* `**` is exponentiation (e.g. `2 ** 3` returns $2^3$),
* `//` is integer division,
* `%` is modulus, e.g. `38 % 5` returns the remainder of `38 // 5`.


---

## Exercise 3

The problem is that the division is computed before the sums, so the first calculation is actually

$$
2 + \frac{5}{5} + 7 = 2 + 1 + 7 = 10.
$$

Adding spaces around the `/` doesn't matter. To solve the problem, add brackets around the sums:

```python
print((2+5) / (5+7))
print(7/12)
```

---

## Exercise 5

```python
my_string = 'Some text characters of my choice.'
m = 6
n = 3

print(my_string[m + n - 1])
print(my_string[m - n - 1])
```


---

## Exercise 6

```python
u = 3
v = 5

# One possible way...
same_sign = (u > 0 and v > 0) or (u < 0 and v < 0)
print(same_sign)

# Another way...
same_sign = u * v > 0
print(same_sign)
```


---

## Exercise 7

```python
x = 1.110223024625156e-16

print(x == 0)
print(x + 1 == 1)
```

The value given is (just under) half of what we call $\epsilon$, or **machine epsilon** --- the distance between 1 and the next closest number in double-precision floating-point representation (i.e. using the `float` type).

This means that any real number $a \in [1, 1+\frac{\epsilon}{2})$ has the same floating-point representation as $1$ -- it effectively gets rounded down to 1. Similarly, any real number $b \in [1 + \frac{\epsilon}{2}, 1 + \epsilon]$ has the same representation as $1 + \epsilon$, because it is rounded up. Now, working in finite precision means that we can actually find *the* largest floating-point number `x`$>0$ such that `x`$<\frac{\epsilon}{2}$ -- this is the value given here.

For a simpler example, if we worked with a system which could only represent numbers up to 3 significant digits:

* We could represent the number $1$ and the number $1.01$, but every real number in between would get rounded to the closest of these values.
* This means that, for this system, $\epsilon=0.01$.
* The largest number `x`$>0$ representable in this system such that `x`$<\frac{\epsilon}{2}=0.005$ is `x = 0.00499`, since we can store at most 3 significant digits.

Note that, in this system, there is no difference between the value of `1 + 0.001`, `1 + 0.00136`, `1 + 0.00487`, etc. --- they all evaluate to `1`. We say that [floating-point addition leads to **round-off error**](https://en.wikipedia.org/wiki/Round-off_error#Addition).

---

## Exercise 8

`help(math)` shows that `math.cos()` needs an input in radians, and `math.radians()` can be used to convert an angle from degrees to radians.

```python
import math
print(math.cos(math.radians(119)))
```


---

## Exercise 9

```python
import numpy as np
print(np.sqrt(2) * np.cos(2 * np.pi / 5))
```

