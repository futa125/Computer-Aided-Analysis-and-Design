import abc
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class Function(abc.ABC):
    default_starting_point: npt.NDArray[np.float64]
    minimum_point: npt.NDArray[np.float64]
    minimum_value: float

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        raise NotImplementedError


@dataclass
class F1(Function):
    default_starting_point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([-1.9, 2.0]))
    minimum_point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([1.0, 1.0]))
    minimum_value: float = 0.0

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        if x.shape != (2,):
            raise ValueError(f"invalid shape {x.shape} for input x")

        x1: float
        x2: float
        x1, x2 = x

        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


@dataclass
class F2(Function):
    default_starting_point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.1, 0.3]))
    minimum_point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([4.0, 2.0]))
    minimum_value: float = 0.0

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        if x.shape != (2,):
            raise ValueError(f"invalid shape {x.shape} for input x")

        x1: float
        x2: float
        x1, x2 = x

        return (x1 - 4) ** 2 + 4 * (x2 - 2) ** 2


@dataclass
class F3(Function):
    default_starting_point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0]))
    minimum_point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([2.0, -3.0]))
    minimum_value: float = 0.0

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        if x.shape != (2,):
            raise ValueError(f"invalid shape {x.shape} for input x")

        x1: float
        x2: float
        x1, x2 = x

        return (x1 - 2) ** 2 + (x2 + 3) ** 2


@dataclass
class F4(Function):
    default_starting_point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0]))
    minimum_point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([3.0, 0.0]))
    minimum_value: float = 0.0

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        if x.shape != (2,):
            raise ValueError(f"invalid shape {x.shape} for input x")

        x1: float
        x2: float
        x1, x2 = x

        return (x1 - 3) ** 2 + x2 ** 2
