from __future__ import annotations

import dataclasses
import logging
from typing import Tuple, Optional, List

import numpy as np

from lab2.functions import BaseFunction, Function1, Function2, Function3, Function4, Function6
from lab2.optimisations.coordinate_search import coordinate_search
from lab2.optimisations.golden_ratio import golden_ration
from lab2.optimisations.hooke_jevees import hooke_jevees
from lab2.optimisations.simplex import nelder_mead
from lab2.types import Point


@dataclasses.dataclass
class FunctionTask1(BaseFunction):
    def function(self: FunctionTask1, p: Point) -> float:
        if p.size != 1:
            raise ValueError("only 1D values supported")

        x = float(p[0])

        return (x - 3) ** 2


def output(x0: Point, f: BaseFunction, step: float = 1.0) -> Tuple[Optional[int], int, int, int]:
    i1 = None
    if x0.size == 1:
        res = golden_ration(starting_point=x0.copy(), starting_interval=None, f=f)
        print(f"Golden ration:\nMinimum - {res}\nIterations - {f.count}\n")
        i1 = f.count
        f.reset()

    res = coordinate_search(starting_point=x0.copy(), f=f)
    print(f"Coordinate search:\nMinimum - {res}\nIterations - {f.count}\n")
    i2 = f.count
    f.reset()

    res = nelder_mead(starting_point=x0.copy(), f=f, step=step)
    print(f"Nelder-Mead:\nMinimum - {res}\nIterations - {f.count}\n")
    i3 = f.count
    f.reset()

    res = hooke_jevees(starting_point=x0.copy(), f=f)
    print(f"Hooke-Jevees:\nMinimum - {res}\nIterations - {f.count}\n")
    i4 = f.count
    f.reset()

    return i1, i2, i3, i4


def task1() -> None:
    print("# Task 1")
    x0 = np.array([10.0])
    t1 = FunctionTask1()

    output(x0, t1, step=0.5)


def task2() -> None:
    print("# Task 2")

    rows: List[str] = []

    f1 = Function1()
    f2 = Function2()
    f3 = Function3()
    f4 = Function4()

    x0 = np.array([-1.9, 2.0])
    i1, i2, i3, i4 = output(x0, f1)
    rows.append(f"#    Function 1: {i2:<20}{i3:<20}{i4:<20}")

    x0 = np.array([0.1, 0.3])
    i1, i2, i3, i4 = output(x0, f2)
    rows.append(f"#    Function 2: {i2:<20}{i3:<20}{i4:<20}")

    x0 = np.zeros(3)
    i1, i2, i3, i4 = output(x0, f3)
    rows.append(f"#    Function 3: {i2:<20}{i3:<20}{i4:<20}")

    x0 = np.array([5.1, 1.1])
    i1, i2, i3, i4 = output(x0, f4)
    rows.append(f"#    Function 4: {i2:<20}{i3:<20}{i4:<20}")

    print(f"# Optimisations: {'Coordinate search':<20}{'Nelder-Mead':<20}{'Hooke-Jevees':<20}")
    print("\n".join(rows))
    print()


def task3() -> None:
    print("# Task 3")
    x0 = np.array([5.0, 5.0])
    f = Function4()

    res = nelder_mead(starting_point=x0.copy(), f=f)
    print(f"Nelder-Mead:\nMinimum - {res}\nIterations - {f.count}\n")
    f.reset()

    res = hooke_jevees(starting_point=x0.copy(), f=f)
    print(f"Hooke-Jevees:\nMinimum - {res}\nIterations - {f.count}\n")
    f.reset()


def task4() -> None:
    print("# Task 4")

    x0 = np.array([0.5, 0.5])
    f = Function1()

    for i in range(1, 20 + 1):
        res = nelder_mead(starting_point=x0.copy(), f=f, step=i)
        print(f"Nelder-Mead (step={i}, x0={x0}):\nMinimum - {res}\nIterations - {f.count}\n")
        f.reset()

    x0 = np.array([20.0, 20.0])
    for i in range(1, 20 + 1):
        res = nelder_mead(starting_point=x0.copy(), f=f, step=i)
        print(f"Nelder-Mead (step={i}, x0={x0}):\nMinimum - {res}\nIterations - {f.count}\n")
        f.reset()


def task5() -> None:
    print("# Task 5")

    f = Function6()

    while True:
        x0 = np.random.uniform(low=-50.0, high=50.0, size=(2,))
        res = nelder_mead(starting_point=x0.copy(), f=f)

        e = 10e-4
        if abs(f(res)) < e:
            print(f"Nelder-Mead (x0={x0}):\nMinimum - {res}\nIterations - {f.count}\nf(minimum)={f(res)}\n")
            f.reset()

            break


def main() -> None:
    task1()
    task2()
    task3()
    task4()
    task5()


if __name__ == "__main__":
    main()
