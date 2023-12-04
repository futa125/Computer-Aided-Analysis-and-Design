from __future__ import annotations

import abc
import dataclasses
from typing import Tuple, Dict

import numpy as np

from lab2.types import Point
from lab3.types import Vector, Matrix


@dataclasses.dataclass
class BaseFunction(abc.ABC):
    call_count: int = dataclasses.field(default=0, init=False)
    gradient_calculation_count: int = dataclasses.field(default=0, init=False)
    hessian_matrix_calculation_count: int = dataclasses.field(default=0, init=False)

    cache: Dict[Tuple[float, ...], float] = dataclasses.field(default_factory=dict, init=False)

    def __call__(self: BaseFunction, p: Point) -> float:
        hashable_point = tuple(p)

        if hashable_point in self.cache:
            return self.cache[hashable_point]

        value = self.function(p)
        self.cache[hashable_point] = value
        self.call_count += 1

        return value

    def reset_cache(self: BaseFunction) -> None:
        self.call_count = 0
        self.cache = {}

    def function(self: BaseFunction, p: Point) -> float:
        raise NotImplementedError

    def gradient(self: BaseFunction, p: Point) -> Vector:
        raise NotImplementedError

    def hessian_matrix(self: BaseFunction, p: Point) -> Matrix:
        raise NotImplementedError

    def starting_point(self: BaseFunction) -> Point:
        raise NotImplementedError

    def minimum(self: BaseFunction) -> (Point, float):
        raise NotImplementedError


@dataclasses.dataclass
class Function1(BaseFunction):
    def function(self: Function1, p: Point) -> float:
        if p.size != 2:
            raise ValueError("only 2D values supported")

        x1 = float(p[0])
        x2 = float(p[1])

        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    def gradient(self: BaseFunction, p: Point) -> Vector:
        x1 = float(p[0])
        x2 = float(p[1])

        dx1 = 2 * (200 * x1 ** 3 - 200 * x1 * x2 + x1 - 1)
        dx2 = 200 * (x2 - x1 ** 2)

        return np.array([dx1, dx2])

    def hessian_matrix(self: BaseFunction, p: Point) -> Matrix:
        x1 = float(p[0])
        x2 = float(p[1])

        dx1dx1 = 1200 * x1 ** 2 - 400 * x2 + 2
        dx2dx2 = 200
        dx1dx2 = -400 * x1
        dx2dx1 = -400 * x1

        return np.array([[dx1dx1, dx1dx2], [dx2dx1, dx2dx2]])

    def starting_point(self: BaseFunction) -> Point:
        return np.array([-1.9, 2.0])

    def minimum(self: BaseFunction) -> (Point, float):
        return np.array([1, 1]), 0


@dataclasses.dataclass
class Function2(BaseFunction):
    def function(self: Function2, p: Point) -> float:
        if p.size != 2:
            raise ValueError("only 2D values supported")

        x1 = float(p[0])
        x2 = float(p[1])

        return (x1 - 4) ** 2 + 4 * (x2 - 2) ** 2

    def gradient(self: BaseFunction, p: Point) -> Vector:
        x1 = float(p[0])
        x2 = float(p[1])

        dx1 = 2 * (x1 - 4)
        dx2 = 8 * (x2 - 2)

        return np.array([dx1, dx2])

    def hessian_matrix(self: BaseFunction, p: Point) -> Matrix:
        dx1dx1 = 2
        dx2dx2 = 8
        dx1dx2 = 0
        dx2dx1 = 0

        return np.array([[dx1dx1, dx1dx2], [dx2dx1, dx2dx2]])

    def starting_point(self: BaseFunction) -> Point:
        return np.array([0.1, 0.3])

    def minimum(self: BaseFunction) -> (Point, float):
        return np.array([4, 2]), 0


@dataclasses.dataclass
class Function3(BaseFunction):
    def function(self: Function3, p: Point) -> float:
        if p.size != 2:
            raise ValueError("only 2D values supported")

        x1 = float(p[0])
        x2 = float(p[1])

        return (x1 - 2) ** 2 + (x2 + 3) ** 2

    def gradient(self: BaseFunction, p: Point) -> Vector:
        x1 = float(p[0])
        x2 = float(p[1])

        dx1 = 2 * (x1 - 2)
        dx2 = 2 * (x2 + 3)

        return np.array([dx1, dx2])

    def hessian_matrix(self: BaseFunction, p: Point) -> Matrix:
        dx1dx1 = 2
        dx2dx2 = 2
        dx1dx2 = 0
        dx2dx1 = 0

        return np.array([[dx1dx1, dx1dx2], [dx2dx1, dx2dx2]])

    def starting_point(self: BaseFunction) -> Point:
        return np.array([0.0, 0.0])

    def minimum(self: BaseFunction) -> (Point, float):
        return np.array([2.0, -3.0]), 0


@dataclasses.dataclass
class Function4(BaseFunction):
    def function(self: Function4, p: Point) -> float:
        if p.size != 2:
            raise ValueError("only 2D values supported")

        x1 = float(p[0])
        x2 = float(p[1])

        return 0.25 * x1 ** 4 - x1 ** 2 + 2 * x1 + (x2 - 1) ** 2

    def starting_point(self: BaseFunction) -> Point:
        raise NotImplementedError

    def minimum(self: BaseFunction) -> (Point, float):
        raise NotImplementedError
