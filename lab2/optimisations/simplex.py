import math
from typing import List

import numpy as np

from lab2.types import Point, ObjectiveFunction


def nelder_mead(
        starting_point: Point,
        f: ObjectiveFunction,
        step=1.0, alfa=1.0, beta=0.5, gamma=2.0, sigma=0.5, e=10e-6,
        logging_enabled: bool = False
):
    simplex_points: List[Point] = [starting_point]

    for i in range(starting_point.size):
        new_point = starting_point.copy()
        new_point[i] += step
        simplex_points.append(new_point)

    while True:
        function_values = []
        for point in simplex_points:
            function_values.append(f(point))

        h, l = np.argmax(function_values), np.argmin(function_values)

        xc = np.mean([point for i, point in enumerate(simplex_points) if i != h], axis=0)
        xr = (1 + alfa) * xc - alfa * simplex_points[h]

        if logging_enabled:
            print(f"centroid={xc}, f(centroid)={f(xc)}")

        if f(xr) < f(simplex_points[l]):
            xe = (1 - gamma) * xc + gamma * xr

            if f(xe) < f(simplex_points[l]):
                simplex_points[h] = xe
            else:
                simplex_points[h] = xr

        else:
            if all(f(xr) > f(simplex_points[j]) for j in range(len(simplex_points)) if j != h):
                if f(xr) < f(simplex_points[h]):
                    simplex_points[h] = xr

                xk = (1 - beta) * xc + beta * simplex_points[h]
                if f(xk) < f(simplex_points[h]):
                    simplex_points[h] = xk

                else:
                    for i in range(len(simplex_points)):
                        if i == l:
                            continue

                        simplex_points[i] = sigma * (simplex_points[i] + simplex_points[l])

            else:
                simplex_points[h] = xr

        value = 0
        for point in simplex_points:
            value += (f(point) - f(xc)) ** 2

        value *= 1 / 2
        value = math.sqrt(value)

        if value < e:
            return xc
