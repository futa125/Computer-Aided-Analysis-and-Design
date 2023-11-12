from __future__ import annotations

import abc
import dataclasses
from typing import Tuple, Dict

import numpy as np

from lab2.cache import point_cache
from lab2.coordinate_search import coordinate_search
from lab2.golden_ratio import golden_ration
from lab2.hooke_jevees import hooke_jevees
from lab2.types import Point


@dataclasses.dataclass
class BaseFunc(abc.ABC):
    count: int = dataclasses.field(default=0, init=False)
    cache: Dict[Tuple[float, ...], Point] = dataclasses.field(default_factory=dict, init=False)

    def __call__(self: BaseFunc, p: Point) -> Point:
        hashable_point = tuple(p)

        if hashable_point in self.cache:
            return self.cache[hashable_point]

        value = self.function(p)
        self.cache[hashable_point] = value
        self.count += 1

        return value

    def reset(self: BaseFunc) -> None:
        self.count = 0
        self.cache = {}

    def function(self: BaseFunc, p: Point) -> Point:
        raise NotImplementedError


@dataclasses.dataclass
class F1(BaseFunc):
    def function(self: F1, p: Point) -> Point:
        x = p[0]

        return (x - 3) ** 2


@dataclasses.dataclass
class FTest(BaseFunc):
    def function(self: FTest, p: Point) -> Point:
        self.count += 1

        x1 = p[0]
        x2 = p[1]

        return (x1 - 4) ** 2 + 4 * (x2 - 2) ** 2


def main() -> None:
    # Task 1
    x0 = np.array([10])
    f1 = F1()

    res = golden_ration(starting_point=x0, starting_interval=None, f=f1)
    print(f"Golden ration:\nMinimum - {res}\nIterations - {f1.count}\n")
    f1.reset()

    res = coordinate_search(starting_point=x0, f=f1)
    print(f"Coordinate search:\nMinimum - {res}\nIterations - {f1.count}\n")
    f1.reset()

    res = hooke_jevees(starting_point=x0, f=f1)
    print(f"Hooke-Jevees:\nMinimum - {res}\nIterations - {f1.count}\n")
    f1.reset()

    # Test

    x0 = np.array([0.1, 0.3])
    ft = FTest()

    res = coordinate_search(starting_point=x0, f=ft)
    print(f"Coordinate search:\nMinimum - {res}\nIterations - {ft.count}\n")
    ft.reset()

    res = hooke_jevees(starting_point=x0, f=ft)
    print(f"Hooke-Jevees:\nMinimum - {res}\nIterations - {ft.count}\n")
    ft.reset()


if __name__ == "__main__":
    main()
