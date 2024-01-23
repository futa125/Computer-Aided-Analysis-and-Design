from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

Predictor = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float, float, Callable],
    npt.NDArray[np.float64],
]


def euler_predictor(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
        t: float,
        t_step: float,
        r: Optional[Callable[[float], npt.NDArray[np.float64]]] = None,
) -> npt.NDArray[np.float64]:
    if r is not None:
        return x + t_step * (np.dot(a, x) + np.dot(b, r(t + t_step)))

    return x + t_step * (np.dot(a, x))
