from typing import Callable, Dict, Tuple, Type

from lab2.types import Point


def point_cache(function: Callable[[Type, Point], Point]):
    cache: Dict[Tuple[float, ...], Point] = {}

    def wrapper(self: Type, point: Point):
        hashable_point = tuple(point)

        if hashable_point in cache:
            return cache[hashable_point]

        value = function(self, point)
        cache[hashable_point] = value

        return value

    return wrapper
