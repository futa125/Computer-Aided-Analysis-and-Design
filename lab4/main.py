from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from lab4.functions import F1, F2, F4, Function
from lab4.optimisations import NumberRange, box, mixed


def task_1() -> None:
    print("# Task 1")

    def _constraint_1(x: npt.NDArray[np.float64]) -> float:
        x1, x2 = x

        return x2 - x1

    def _constraint_2(x: npt.NDArray[np.float64]) -> float:
        x1, _ = x

        return 2 - x1

    explicit_constraints = NumberRange(lower_bound=-100, upper_bound=100)
    implicit_constraints: Tuple[Callable[[npt.NDArray[np.float64]], float], ...] = (_constraint_1, _constraint_2)

    f: Function = F1()

    while True:
        minimum: Optional[npt.NDArray[np.float64]] = box(
            f, f.default_starting_point, explicit_constraints, implicit_constraints,
        )

        if minimum is not None:
            print("Box + Function 1 + Constraints")
            print(f"Minimum: {minimum}, value: {f(minimum)}")
            break

    f: Function = F1()

    while True:
        minimum: Optional[npt.NDArray[np.float64]] = box(
            f, f.default_starting_point,  # explicit_constraints, implicit_constraints,
        )

        if minimum is not None:
            print("Box + Function 1 + No constraints")
            print(f"Minimum: {minimum}, value: {f(minimum)}")
            break

    f: Function = F2()

    while True:
        minimum: Optional[npt.NDArray[np.float64]] = box(
            f, f.default_starting_point, explicit_constraints, implicit_constraints,
        )

        if minimum is not None:
            print("Box + Function 2 + Constraints")
            print(f"Minimum: {minimum}, value: {f(minimum)}")
            break

    f: Function = F2()

    while True:
        minimum: Optional[npt.NDArray[np.float64]] = box(
            f, f.default_starting_point,  # explicit_constraints, implicit_constraints,
        )

        if minimum is not None:
            print("Box + Function 2 + No constraints")
            print(f"Minimum: {minimum}, value: {f(minimum)}")
            break

    print()


def task_2() -> None:
    print("# Task 2")

    def _constraint_1(x: npt.NDArray[np.float64]) -> float:
        x1, x2 = x

        return x2 - x1

    def _constraint_2(x: npt.NDArray[np.float64]) -> float:
        x1, _ = x

        return 2 - x1

    implicit_constraints: Tuple[Callable[[npt.NDArray[np.float64]], float], ...] = (_constraint_1, _constraint_2)

    f: Function = F1()

    minimum: npt.NDArray[np.float64] = mixed(f, f.default_starting_point, implicit_constraints)

    print(f"Function 1 + Starting point {f.default_starting_point}")
    print(f"Minimum: {minimum}, value: {f(minimum)}")

    f: Function = F2()

    minimum: npt.NDArray[np.float64] = mixed(f, f.default_starting_point, implicit_constraints)

    print(f"Function 2 + Starting point {f.default_starting_point}")
    print(f"Minimum: {minimum}, value: {f(minimum)}")

    print()


def task_3() -> None:
    print("# Task 3")

    def _constraint_1(x: npt.NDArray[np.float64]) -> float:
        x1, x2 = x

        return x2 - x1

    def _constraint_2(x: npt.NDArray[np.float64]) -> float:
        x1, _ = x

        return 2 - x1

    def _constraint_3(x: npt.NDArray[np.float64]) -> float:
        _, x2 = x

        return x2 - 1

    implicit_constraints: Tuple[Callable[[npt.NDArray[np.float64]], float], ...] = (
        _constraint_1, _constraint_2, _constraint_3,
    )

    f: Function = F4()

    starting_point: npt.NDArray[np.float64] = np.array([5.0, 5.0])
    minimum: npt.NDArray[np.float64] = mixed(f, starting_point, implicit_constraints)

    print(f"Function 4 + Starting point {starting_point}")
    print(f"Minimum: {minimum}, value: {f(minimum)}")

    print()


def main() -> None:
    task_1()
    task_2()
    task_3()


if __name__ == "__main__":
    main()
