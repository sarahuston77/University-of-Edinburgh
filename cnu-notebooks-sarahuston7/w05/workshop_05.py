def bubble_sort(list1: list[int]) -> list[int]:
    """Returns a sorted list using bubble method i.e. swap method."""
    n = len(list1)
    swapped = True
    while swapped:
        swapped = False
        for idx in range(1, n):
            if list1[idx - 1] > list1[idx]:
                list1[idx - 1], list1[idx] = list1[idx], list1[idx - 1]
                swapped = True
    return list1


import numpy as np
import time as t
import random as r
import matplotlib.pyplot as plt


def time_sort(fun):
    """Executes bubble_sort on random lists of size 500-300, increments 500 and times/plots it."""

    random_list = [*range(500, 3001, 500)]
    size_list: list[int] = []
    time_list: list[int] = []  

    i = 0
    while i < 6: 
        idx: int = r.randint(0, len(random_list) - 1)
        size_list.append(random_list[idx])
        t0 = t.time()
        fun([r.randint(0, 10000) for f in range(0, random_list[idx])])
        time_list.append(t.time() - t0)
        i += 1
    return size_list, time_list


def merge_sort(list2: list[int]) -> list:
    """Uses recursion to sort lists."""
    
    a_list: list[int] = list2[:len(list2) // 2]
    b_list: list[int] = list2[len(list2) // 2:]
    
    if len(a_list) != 1:
        a_list = merge_sort(a_list)

    if len(b_list) != 1:
        b_list = merge_sort(b_list)
    
    return merge(a_list, b_list)


def merge(list3: list[int], list4: list[int]) -> list:
    """Merge two lists in sorted order."""
    new_list: list[int] = []
    print(f"this is list1 of merge: {list3}")
    print(f"this is list2 of merge: {list4}")
    while len(list3) > 0 and len(list4) > 0:
        if list3[0] > list4[0]:
            new_list.append(list4[0])
            list4.pop(0)
        else:
            new_list.append(list3[0])
            list3.pop(0)
    
    if len(list4) > 0:
        new_list += list4
    if len(list3) > 0:
        new_list += list3
    print(f"this is new_list: {new_list}")
    return new_list
 

hey = [0, 10, 8, 3, 2, 1]

# size_list, time_list = time_sort(merge_sort)
# figure, ax = plt.subplots()
# ax.plot(size_list, time_list)
# plt.show()



def quicksort(listy: list[int] = []) -> list:
    """Quicksort method of sorting lists."""
    
    if len(listy) <= 1:
        return listy
    
    m = int(len(listy) * r.random())
    before: list[int] = []
    after: list[int] = []
    equal: list[int] = [listy[m]]
    for idx in range(len(listy)):
        if listy[idx] > listy[m]:
            after.append(listy[idx])
        elif listy[idx] < listy[m]:  
            before.append(listy[idx])
    return quicksort(before) + equal + quicksort(after)

print(quicksort(hey))