from typing import Callable

import numpy as np
import numpy.typing as npt

Corrector = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float, float, Callable],
    npt.NDArray[np.float64],
]


def reverse_euler_corrector(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        x_current: npt.NDArray[np.float64],
        x_next: npt.NDArray[np.float64],
        t: float,
        t_step: float,
        r: Callable = None,
) -> npt.NDArray[np.float64]:
    if r is not None:
        return x_current + t_step * (np.dot(a, x_next) + np.dot(b, r(t + t_step)))

    return x_current + t_step * np.dot(a, x_next)


def trapezoidal_corrector(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        x_current: npt.NDArray[np.float64],
        x_next: npt.NDArray[np.float64],
        t: float,
        t_step: float,
        r: Callable = None,
) -> npt.NDArray[np.float64]:
    if r is not None:
        return x_current + t_step / 2 * ((np.dot(a, x_current) + np.dot(b, r(t))) + np.dot(a, x_next) + np.dot(b, r(t + t_step)))

    return x_current + t_step / 2 * (np.dot(a, x_current) + np.dot(a, x_next))
