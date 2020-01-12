from typing import List, Iterable, Union


class Vector(list):
    """A simple vector class"""
    def __init__(self, *values: Iterable) -> None:
        if isinstance(values, list):
            self.values = values 
        else:
            self.values = list(values)
