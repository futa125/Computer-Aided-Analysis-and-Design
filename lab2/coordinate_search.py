import dataclasses

import numpy as np

from lab2.cache import point_cache
from lab2.golden_ratio import golden_ration
from lab2.types import Point, ObjectiveFunction


@dataclasses.dataclass
class MinimiseLambdaFunction:
    x: Point
    i: int
    f: ObjectiveFunction

    def __call__(self, lambda_value: Point) -> Point:
        x = self.x.copy()
        x[self.i] += lambda_value

        return self.f(x)


def coordinate_search(starting_point: Point, f: ObjectiveFunction, e=10e-6):
    x = starting_point

    xs = x.copy()
    for i in range(len(x)):
        x[i] += golden_ration(starting_point=np.array([x[i]]), f=MinimiseLambdaFunction(x, i, f))

    while (np.abs(x - xs) > e).any():
        xs = x.copy()
        for i in range(len(x)):
            x[i] += golden_ration(starting_point=np.array([x[i]]), f=MinimiseLambdaFunction(x, i, f))

    return x
