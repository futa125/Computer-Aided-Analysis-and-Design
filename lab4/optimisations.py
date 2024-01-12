import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from lab2.optimisations.hooke_jevees import hooke_jevees
from lab4.functions import Function


@dataclass
class NumberRange:
    lower_bound: float
    upper_bound: float

    def __contains__(self, number: float) -> bool:
        return self.lower_bound <= number <= self.upper_bound


def box(
        function: Function,
        starting_point: npt.NDArray[np.float64],
        explicit_constraint: NumberRange = NumberRange(-10, 10),
        implicit_constraint: Tuple[Callable[[npt.NDArray[np.float64]], float], ...] = (),
        alpha: float = 1.3,
        epsilon: float = 10e-6,
) -> Optional[npt.NDArray[np.float64]]:
    if any(value not in explicit_constraint for value in starting_point):
        raise ValueError("starting point doesn't satisfy explicit constraint")

    if any(constraint(starting_point) < 0 for constraint in implicit_constraint):
        raise ValueError("starting point doesn't satisfy implicit constraint")

    n: int = starting_point.size
    xc: npt.NDArray[np.float64] = starting_point.copy()
    simplex: List[npt.NDArray[np.float64]] = [xc]

    for _ in range(2 * n):
        x: npt.NDArray[np.float64] = (
                explicit_constraint.lower_bound +
                np.random.rand(n, ) * (explicit_constraint.upper_bound - explicit_constraint.lower_bound)
        )

        while any(constraint(starting_point) < 0 for constraint in implicit_constraint):
            x: npt.NDArray[np.float64] = 0.5 * (x + xc)

        simplex.append(x)
        xc: npt.NDArray[np.float64] = np.mean(simplex, axis=0)

    while True:
        function_values: List[float] = [function(point) for point in simplex]

        h1: int
        h2: int
        h2, h1 = np.argsort(function_values)[-2:]

        xc: npt.NDArray[np.float64] = np.mean([row for i, row in enumerate(simplex) if i != h1], axis=0)
        xr: npt.NDArray[np.float64] = (1 + alpha) * xc - alpha * simplex[h1]
        xr: npt.NDArray[np.float64] = xr.clip(explicit_constraint.lower_bound, explicit_constraint.upper_bound)

        if np.isclose(xc, xr).all():
            return None

        while any(constraint(xr) < 0 for constraint in implicit_constraint):
            if np.isclose(xr, xc).all():
                return None

            xr: npt.NDArray[np.float64] = 0.5 * (xr + xc)

        if function(xr) > function(simplex[h2]):
            xr: npt.NDArray[np.float64] = 0.5 * (xr + xc)

        simplex[h1] = xr

        mean_squared_difference: float = 0
        for row in simplex:
            mean_squared_difference += (function(row) - function(xc)) ** 2

        mean_squared_difference /= n

        if math.sqrt(mean_squared_difference) < epsilon:
            return xc


def mixed(
        function: Function,
        starting_point: npt.NDArray[np.float64],
        inequality_constraints: Tuple[Callable[[npt.NDArray[np.float64]], float], ...] = (),
        equality_constraints: Tuple[Callable[[npt.NDArray[np.float64]], float], ...] = (),
        t: int = 10,
        epsilon: float = 10e-6,
) -> npt.NDArray[np.float64]:
    x: npt.NDArray[np.float64] = starting_point.copy()
    if any(constraint(x) < 0 for constraint in inequality_constraints):
        x: npt.NDArray[np.float64] = hooke_jevees(starting_point, _FindInnerPoint(inequality_constraints))

    while True:
        prev_x: npt.NDArray[np.float64] = x.copy()
        x: npt.NDArray[np.float64] = hooke_jevees(
            x, _OptimiseWithConstraints(function, inequality_constraints, equality_constraints, t),
        )

        if np.isclose(prev_x, x, atol=epsilon).all():
            return x

        t *= 10


@dataclass
class _FindInnerPoint:
    constraints: Tuple[Callable[[npt.NDArray[np.float64]], float], ...]

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        return -sum(constraint(x) for constraint in self.constraints if constraint(x) < 0)


@dataclass
class _OptimiseWithConstraints:
    function: Callable[[npt.NDArray[np.float64]], float]
    inequality_constraints: Tuple[Callable[[npt.NDArray[np.float64]], float], ...]
    equality_constraints: Tuple[Callable[[npt.NDArray[np.float64]], float], ...]
    t: int

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        result: float = self.function(x)

        for constraint in self.inequality_constraints:
            if constraint(x) <= 0:
                return math.inf

            result -= (1 / self.t) * math.log(constraint(x))

        for constraint in self.equality_constraints:
            result += self.t * constraint(x) ** 2

        return result
