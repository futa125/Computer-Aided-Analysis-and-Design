import math
from typing import List

import numpy as np

from lab2.types import Point, ObjectiveFunction


def nelder_mead(
        starting_point: Point, f: ObjectiveFunction, step=1.0, alfa=1.0, beta=0.5, gamma=2.0, sigma=0.5, e=10e-6
):
    simplex_points: List[Point] = [starting_point]

    for i in range(len(starting_point)):
        new_point = starting_point.copy()
        new_point[i] += step
        simplex_points.append(new_point)

    while True:
        function_values = []
        for point in simplex_points:
            function_values.append(f(point))

        h, l = np.argmax(function_values), np.argmin(function_values)

        xh = simplex_points[h]
        xc = np.mean([point for i, point in enumerate(simplex_points) if i != h], axis=0)
        xr = (1 + alfa) * xc - alfa * xh

        if f(xr) < function_values[l]:
            xe = (1 - gamma) * xc + gamma * xr

            if f(xe) < function_values[l]:
                simplex_points[h] = xe
                function_values[h] = f(xe)
            else:
                simplex_points[h] = xr
                function_values[h] = f(xr)

        else:
            if all(f(xr) > function_values[j] for j in range(starting_point.size) if j != h):
                if f(xr) < function_values[h]:
                    simplex_points[h] = xr
                    function_values[h] = f(xr)

                xk = (1 - beta) * xc + beta * xh
                if f(xk) < function_values[h]:
                    simplex_points[h] = xk
                    function_values[h] = f(xk)

                else:
                    for i, point in enumerate(simplex_points):
                        if i == l:
                            continue

                        simplex_points[i] = sigma * (simplex_points[i] + simplex_points[l])
                        function_values[i] = f(point)

            else:
                simplex_points[h] = xr
                function_values[h] = f(xr)

        value = 0
        for fxi in function_values:
            value += (fxi - f(xc)) ** 2

        value *= 1 / 2
        value = math.sqrt(value)

        if value < e:
            return xc
