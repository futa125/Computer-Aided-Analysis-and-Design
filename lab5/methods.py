from typing import Callable, List, Optional

import numpy as np
import numpy.typing as npt

from lab5.corrector import Corrector
from lab5.predictor import Predictor


def runge_kutta(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        starting_state: npt.NDArray[np.float64],
        t_start: float,
        t_max: float,
        t_step: float,
        r: Optional[Callable[[float], npt.NDArray[np.float64]]] = None,
) -> List[npt.NDArray[np.float64]]:
    x: List[npt.NDArray[np.float64]] = [starting_state]
    values = np.linspace(t_start + t_step, t_max, int((t_max - t_start) / t_step) - 1)

    i: int
    t: float
    for i, t in enumerate(values):
        if r is not None:
            m1 = np.dot(a, x[i]) + np.dot(b, r(t))
            m2 = np.dot(a, x[i] + t_step / 2 * m1) + np.dot(b, r(t + t_step / 2))
            m3 = np.dot(a, x[i] + t_step / 2 * m2) + np.dot(b, r(t + t_step / 2))
            m4 = np.dot(a, x[i] + t_step * m3) + np.dot(b, r(t + t_step))

        else:
            m1 = np.dot(a, x[i])
            m2 = np.dot(a, x[i] + t_step / 2 * m1)
            m3 = np.dot(a, x[i] + t_step / 2 * m2)
            m4 = np.dot(a, x[i] + t_step * m3)

        x.append(x[i] + t_step / 6 * (m1 + 2 * m2 + 2 * m3 + m4))

    return x


def trapezoidal(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        starting_state: npt.NDArray[np.float64],
        t_start: float,
        t_max: float,
        t_step: float,
        r_func: Optional[Callable[[float], npt.NDArray[np.float64]]] = None,
) -> List[npt.NDArray[np.float64]]:
    x: List[npt.NDArray[np.float64]] = [starting_state]
    values = np.linspace(t_start + t_step, t_max, int((t_max - t_start) / t_step) - 1)

    p = np.linalg.inv(np.identity(2) - np.dot(a, t_step / 2))
    r = np.dot(p, np.identity(2) + np.dot(a, t_step / 2))
    s = np.dot(np.dot(p, t_step / 2), b)

    i: int
    t: float
    for i, t in enumerate(values):
        if r_func is not None:
            x.append(np.dot(r, x[i]) + np.dot(s, r_func(t) + r_func(t + t_step)))
        else:
            x.append(np.dot(r, x[i]))

    return x


def euler(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        starting_state: npt.NDArray[np.float64],
        t_start: float,
        t_max: float,
        t_step: float,
        r_func: Optional[Callable[[float], npt.NDArray[np.float64]]] = None,
) -> List[npt.NDArray[np.float64]]:
    x_values: List[npt.NDArray[np.float64]] = [starting_state]
    t_values = np.linspace(t_start + t_step, t_max, int((t_max - t_start) / t_step) - 1)

    dx: List[npt.NDArray[np.float64]] = []
    if r_func is not None:
        dx.append(np.dot(a, x_values[0]) + np.dot(b, r_func(t_start)))
    else:
        dx.append(np.dot(a, x_values[0]))

    i: int
    t: float
    for i, t in enumerate(t_values):
        x_values.append(x_values[i] + t_step * dx[i])

        if r_func is not None:
            dx.append(np.dot(a, x_values[i + 1]) + np.dot(b, r_func(t)))
        else:
            dx.append(np.dot(a, x_values[i + 1]))

    return x_values


def reverse_euler(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        starting_state: npt.NDArray[np.float64],
        t_start: float,
        t_max: float,
        t_step: float,
        r_func: Optional[Callable[[float], npt.NDArray[np.float64]]] = None,
) -> List[npt.NDArray[np.float64]]:
    x: List[npt.NDArray[np.float64]] = [starting_state]
    values = np.linspace(t_start + t_step, t_max, int((t_max - t_start) / t_step) - 1)

    p = np.linalg.inv(np.identity(2) - np.dot(a, t_step))
    q = np.dot(np.dot(p, t_step), b)

    i: int
    t: float
    for i, t in enumerate(values):
        if r_func is not None:
            x.append(np.dot(p, x[i]) + np.dot(q, r_func(t + t_step)))
        else:
            x.append(np.dot(p, x[i]))

    return x


def predictor_corrector(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        starting_state: npt.NDArray[np.float64],
        t_start: float,
        t_max: float,
        t_step: float,
        predictor: Predictor,
        corrector: Corrector,
        corrector_iterations: int,
        r_func: Optional[Callable[[float], npt.NDArray[np.float64]]] = None,
) -> List[npt.NDArray[np.float64]]:
    x: List[npt.NDArray[np.float64]] = [starting_state]
    values = np.linspace(t_start + t_step, t_max, int((t_max - t_start) / t_step) - 1)

    i: int
    t: float
    for i, t in enumerate(values):
        value = predictor(a, b, x[i], t, t_step, r_func)
        for j in range(corrector_iterations):
            value = corrector(a, b, x[i], value, t, t_step, r_func)

        x.append(value)

    return x
