from __future__ import annotations

import dataclasses

import numpy as np

from lab2.functions import BaseFunc, Function1, Function2, Function3
from lab2.optimisations.coordinate_search import coordinate_search
from lab2.optimisations.golden_ratio import golden_ration
from lab2.optimisations.hooke_jevees import hooke_jevees
from lab2.optimisations.simplex import nelder_mead
from lab2.types import Point


@dataclasses.dataclass
class Task1(BaseFunc):
    def function(self: Task1, p: Point) -> float:
        if p.size != 1:
            raise ValueError("only 1D values supported")

        x = float(p[0])

        return (x - 3) ** 2


def output(x0: Point, f: BaseFunc) -> None:
    if x0.size == 1:
        res = golden_ration(starting_point=x0.copy(), starting_interval=None, f=f)
        print(f"Golden ration:\nMinimum - {res}\nIterations - {f.count}\n")
        f.reset()

    res = coordinate_search(starting_point=x0.copy(), f=f)
    print(f"Coordinate search:\nMinimum - {res}\nIterations - {f.count}\n")
    f.reset()

    res = nelder_mead(starting_point=x0.copy(), f=f)
    print(f"Nelder-Mead:\nMinimum - {res}\nIterations - {f.count}\n")
    f.reset()

    res = hooke_jevees(starting_point=x0.copy(), f=f)
    print(f"Hooke-Jevees:\nMinimum - {res}\nIterations - {f.count}\n")
    f.reset()


def main() -> None:
    # Task 1
    print(f"# Task 1")
    x0 = np.array([10.0])
    t1 = Task1()

    output(x0, t1)

    # Function 1
    print(f"# Function 1")
    x0 = np.array([-1.9, 2.0])
    f1 = Function1()

    output(x0, f1)

    # Function 2
    print(f"# Function 2")
    x0 = np.array([0.1, 0.3])
    f2 = Function2()

    output(x0, f2)

    # Function 3
    print(f"# Function 3")
    x0 = np.zeros(10)
    f3 = Function3()

    output(x0, f3)


if __name__ == "__main__":
    main()
