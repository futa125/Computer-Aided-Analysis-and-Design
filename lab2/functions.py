from __future__ import annotations

import abc
import dataclasses
import math
from typing import Tuple, Dict

import numpy as np

from lab2.types import Point


@dataclasses.dataclass
class BaseFunction(abc.ABC):
    count: int = dataclasses.field(default=0, init=False)
    cache: Dict[Tuple[float, ...], float] = dataclasses.field(default_factory=dict, init=False)

    def __call__(self: BaseFunction, p: Point) -> float:
        hashable_point = tuple(p)

        if hashable_point in self.cache:
            return self.cache[hashable_point]

        value = self.function(p)
        self.cache[hashable_point] = value
        self.count += 1

        return value

    def reset(self: BaseFunction) -> None:
        self.count = 0
        self.cache = {}

    def function(self: BaseFunction, p: Point) -> float:
        raise NotImplementedError


@dataclasses.dataclass
class Function1(BaseFunction):
    def function(self: Function1, p: Point) -> float:
        if p.size != 2:
            raise ValueError("only 2D values supported")

        x1 = float(p[0])
        x2 = float(p[1])

        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


@dataclasses.dataclass
class Function2(BaseFunction):
    def function(self: Function2, p: Point) -> float:
        if p.size != 2:
            raise ValueError("only 2D values supported")

        x1 = float(p[0])
        x2 = float(p[1])

        return (x1 - 4) ** 2 + 4 * (x2 - 2) ** 2


@dataclasses.dataclass
class Function3(BaseFunction):
    def function(self: Function3, p: Point) -> float:
        return np.sum(np.square(p - np.arange(1, p.size + 1)))


@dataclasses.dataclass
class Function4(BaseFunction):
    def function(self: Function4, p: Point) -> float:
        if p.size != 2:
            raise ValueError("only 2D values supported")

        x1 = float(p[0])
        x2 = float(p[1])

        return math.fabs((x1 - x2) * (x1 + x2)) + math.sqrt(x1 ** 2 + x2 ** 2)


@dataclasses.dataclass
class Function6(BaseFunction):
    def function(self: Function6, x: Point) -> float:
        return (0.5 + (np.square(np.sin(np.sqrt(np.sum(np.square(x))))) - 0.5) /
                np.square(1 + 0.001 * np.sum(np.square(x))))
