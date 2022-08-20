"""Collection of list utils"""
from typing import Iterable, List


def intersect(lists: Iterable):
    """Intersection between arbitrary number of lists"""
    return sorted(set.intersection(*map(set, lists)), key=lists[0].index)


def argmin(lst: List) -> int:
    """Returns the argmin (index) in a list of numbers."""
    return min(range(len(lst)), key=lambda x: lst[x])


def argmax(lst: List) -> int:
    """Returns the argmax (index) in a list of numbers."""
    return max(range(len(lst)), key=lambda x: lst[x])
