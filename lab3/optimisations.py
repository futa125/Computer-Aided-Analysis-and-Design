import dataclasses
from typing import Set, Tuple

import numpy as np
import numpy.typing as npt
from scipy.linalg import lu_factor, lu_solve

from lab2.optimisations.golden_ratio import golden_ration
from lab3.functions import BaseFunction


@dataclasses.dataclass
class MinimiseLambdaFunction:
    x: npt.NDArray[np.float64]
    step: npt.NDArray[np.float64]
    f: BaseFunction

    def __call__(self, lambda_value: npt.NDArray[np.float64]) -> float:
        return self.f.value(self.x + (lambda_value * self.step))


def gradient_descent(
        starting_point: npt.NDArray[np.float64], f: BaseFunction, e=10e-6, use_golden_ratio=True,
) -> (npt.NDArray[np.float64], float):
    x = starting_point.copy()
    old_value = f.value(x)

    gradient = f.gradient(x)

    visited_points: Set[Tuple[float, ...]] = set()
    divergence_count = 0

    while np.linalg.norm(gradient) > e:
        if tuple(x) in visited_points:
            raise Exception(f"stuck in loop due to repeated occurrence of point {x}")

        visited_points.add(tuple(x))

        if divergence_count > 10:
            raise Exception("failed to find minimum due to divergence")

        if use_golden_ratio:
            min_lambda = golden_ration(MinimiseLambdaFunction(x, gradient, f), starting_point=np.array([0.0]))
            x += min_lambda * gradient
        else:
            x -= gradient

        gradient = f.gradient(x)

        new_value = f.value(x)
        if new_value > old_value:
            divergence_count += 1
        else:
            divergence_count = 0

        old_value = new_value

    return x, old_value


def newton_raphson(
        starting_point: npt.NDArray[np.float64], f: BaseFunction, e=10e-6, use_golden_ratio=True,
) -> (npt.NDArray[np.float64], float):
    x = starting_point.copy()
    old_value = f.value(x)

    min_lambda = 1
    step = lu_solve(lu_factor(f.hessian_matrix(x)), f.gradient(x))

    visited_points: Set[Tuple[float, ...]] = set()
    divergence_count = 0

    while np.linalg.norm(min_lambda * step) > e:
        if tuple(x) in visited_points:
            raise Exception(f"stuck in loop due to repeated occurrence of point {x}")

        visited_points.add(tuple(x))

        if divergence_count > 10:
            raise Exception("failed to find minimum due to divergence")

        if use_golden_ratio:
            min_lambda = golden_ration(MinimiseLambdaFunction(x, step, f), starting_point=np.array([0.0]))
            x += min_lambda * step
        else:
            x -= step

        step = lu_solve(lu_factor(f.hessian_matrix(x)), f.gradient(x))

        new_value = f.value(x)
        if new_value > old_value:
            divergence_count += 1
        else:
            divergence_count = 0

        old_value = new_value

    return x, old_value


def gauss_newton(
        starting_point: npt.NDArray[np.float64], f: BaseFunction, e=10e-6, use_golden_ratio=True,
) -> (npt.NDArray[np.float64], float):
    x = starting_point.copy()
    old_value = f.value(x)

    min_lambda = 1
    jacobian = f.jacobian(x)
    system_of_equations = f.system_of_equations(x)
    a = np.dot(np.transpose(jacobian), jacobian)
    b = np.dot(np.transpose(jacobian), system_of_equations)
    step = lu_solve(lu_factor(a), b)

    divergence_count = 0

    while np.linalg.norm(min_lambda * step) > e:
        if divergence_count > 10:
            raise Exception("failed to find minimum due to divergence")

        if use_golden_ratio:
            min_lambda = golden_ration(MinimiseLambdaFunction(x, step, f), starting_point=np.array([0.0]))
            x += min_lambda * step
        else:
            x -= step

        jacobian = f.jacobian(x)
        system_of_equations = f.system_of_equations(x)
        a = np.dot(np.transpose(jacobian), jacobian)
        b = np.dot(np.transpose(jacobian), system_of_equations)
        step = lu_solve(lu_factor(a), b)

        new_value = f.value(x)
        if new_value > old_value:
            divergence_count += 1
        else:
            divergence_count = 0

        old_value = new_value

    return x, old_value
