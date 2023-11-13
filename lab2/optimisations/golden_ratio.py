import logging
import math
from typing import Optional

import numpy as np

from lab2.types import Point, ObjectiveFunction, Interval

k = 0.5 * (math.sqrt(5) - 1)


def golden_ration(
        f: ObjectiveFunction,
        starting_point: Optional[Point] = None,
        starting_interval: Optional[Interval] = None,
        e: float = 10e-6,
        h: float = 1.0,
        logging_enabled: bool = False,
) -> Point:
    if starting_point.size != 1:
        raise ValueError("golden ratio only support 1-dimensional values")

    a: Point = np.array([0.0])
    b: Point = np.array([0.0])

    if starting_point is not None:
        a, b = unimodal(starting_point, h, f)
    elif starting_interval is not None:
        a, b = starting_interval
    else:
        ValueError("must specify either starting point or starting interval")

    c = b - k * (b - a)
    d = a + k * (b - a)
    fc = f(c)
    fd = f(d)

    if logging_enabled:
        print(f"a={a}, b={b}, c={c}, d={d}")
        print(f"f(a)={f(a)}, f(b)={f(b)}, f(c)={f(c)}, f(d)={f(d)}")

    while (b - a) > e:
        if fc < fd:
            b = d
            d = c
            c = b - k * (b - a)
            fd = fc
            fc = f(c)
        else:
            a = c
            c = d
            d = a + k * (b - a)
            fc = fd
            fd = f(d)

        if logging_enabled:
            print(f"a={a}, b={b}, c={c}, d={d}")
            print(f"f(a)={f(a)}, f(b)={f(b)}, f(c)={f(c)}, f(d)={f(d)}")

    return (a + b) / 2


def unimodal(point: Point, h: float, f: ObjectiveFunction) -> Interval:
    l = point - h
    r = point + h
    m = point
    step = 1

    fm = f(point)
    fl = f(l)
    fr = f(r)

    if fm < fr and fm < fl:
        return l, r

    if fm > fr:
        while fm > fr:
            l = m
            m = r
            fm = fr
            step *= 2
            r = point + h * step
            fr = f(r)

        return l, r

    while fm > fl:
        r = m
        m = l
        fm = fl
        step *= 2
        l = point - h * step
        fl = f(l)

    return l, r
