import dataclasses

import numpy as np

from lab2.optimisations.golden_ratio import golden_ration
from lab2.types import Point, ObjectiveFunction
from lab3.functions import BaseFunction
from lab3.types import Vector


@dataclasses.dataclass
class MinimiseLambdaFunction:
    x: Point
    gradient: Vector
    f: ObjectiveFunction

    def __call__(self, lambda_value: Point) -> float:
        return self.f(self.x + (lambda_value * self.gradient))


def gradient_descent(starting_point: Point, f: BaseFunction, e=10e-6, use_golden_ratio=True) -> (Point, float):
    x = starting_point.copy()
    gradient = f.gradient(x)

    f_value = f(x)
    divergence_count = 0

    while np.linalg.norm(gradient) > e:
        if divergence_count > 10:
            raise Exception("failed to find minimum due to divergence")

        if use_golden_ratio:
            min_lambda = golden_ration(MinimiseLambdaFunction(x, gradient, f), starting_point=np.array([0.0]))
            x += min_lambda * gradient
        else:
            x -= gradient

        gradient = f.gradient(x)

        if f(x) > f_value:
            divergence_count += 1
        else:
            divergence_count = 0

        f_value = f(x)

    return x, f_value

