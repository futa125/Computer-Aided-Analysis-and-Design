from __future__ import annotations

import abc
import dataclasses
from typing import Tuple

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class BaseFunction(abc.ABC):
    starting_point: npt.NDArray[np.float64] = dataclasses.field(init=False)
    minimum: Tuple[npt.NDArray[np.float64], float] = dataclasses.field(init=False)

    call_count: int = dataclasses.field(default=0, init=False)
    gradient_calculation_count: int = dataclasses.field(default=0, init=False)
    hessian_matrix_calculation_count: int = dataclasses.field(default=0, init=False)

    def value(self: BaseFunction, x: npt.NDArray[np.float64]) -> float:
        self.call_count += 1
        return self._value(x)

    def gradient(self: BaseFunction, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.gradient_calculation_count += 1
        return self._gradient(x)

    def hessian_matrix(self: BaseFunction, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.hessian_matrix_calculation_count += 1
        return self._hessian_matrix(x)

    @staticmethod
    def _value(x: npt.NDArray[np.float64]) -> float:
        raise NotImplementedError

    @staticmethod
    def _gradient(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @staticmethod
    def _hessian_matrix(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @staticmethod
    def system_of_equations(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @staticmethod
    def jacobian(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError


@dataclasses.dataclass
class Function1(BaseFunction):
    def __post_init__(self: Function1) -> None:
        self.starting_point: npt.NDArray[np.float64] = np.array([-1.9, 2.0])
        self.minimum: Tuple[npt.NDArray[np.float64], float] = (np.array([1.0, 1.0]), 0.0)

    @staticmethod
    def _value(x: npt.NDArray[np.float64]) -> float:
        x1 = float(x[0])
        x2 = float(x[1])

        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    
    @staticmethod
    def _gradient(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])
        x2 = float(x[1])

        dx1 = 2 * (200 * x1 ** 3 - 200 * x1 * x2 + x1 - 1)
        dx2 = 200 * (x2 - x1 ** 2)

        return np.array([dx1, dx2])

    @staticmethod
    def _hessian_matrix(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])
        x2 = float(x[1])

        dx1dx1 = 1200 * x1 ** 2 - 400 * x2 + 2
        dx2dx2 = 200
        dx1dx2 = -400 * x1
        dx2dx1 = -400 * x1

        return np.array([[dx1dx1, dx1dx2], [dx2dx1, dx2dx2]])

    @staticmethod
    def system_of_equations(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])
        x2 = float(x[1])

        return np.array([x2 - x1 ** 2, 1 - x1])

    @staticmethod
    def jacobian(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])

        return np.array([[-2 * x1, 1], [-1, 0]])


@dataclasses.dataclass
class Function2(BaseFunction):
    def __post_init__(self: Function1) -> None:
        self.starting_point: npt.NDArray[np.float64] = np.array([0.1, 0.3])
        self.minimum: Tuple[npt.NDArray[np.float64], float] = (np.array([4.0, 2.0]), 0.0)

    @staticmethod
    def _value(x: npt.NDArray[np.float64]) -> float:
        x1 = float(x[0])
        x2 = float(x[1])

        return (x1 - 4) ** 2 + 4 * (x2 - 2) ** 2

    @staticmethod
    def _gradient(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])
        x2 = float(x[1])

        dx1 = 2 * (x1 - 4)
        dx2 = 8 * (x2 - 2)

        return np.array([dx1, dx2])

    @staticmethod
    def _hessian_matrix(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        dx1dx1 = 2
        dx2dx2 = 8
        dx1dx2 = 0
        dx2dx1 = 0

        return np.array([[dx1dx1, dx1dx2], [dx2dx1, dx2dx2]])

    @staticmethod
    def system_of_equations(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])
        x2 = float(x[1])

        return np.array([x1 - 4, x2 - 2])

    @staticmethod
    def jacobian(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[1, 0], [0, 1]])


@dataclasses.dataclass
class Function3(BaseFunction):
    def __post_init__(self: Function1) -> None:
        self.starting_point: npt.NDArray[np.float64] = np.array([0.0, 0.0])
        self.minimum: Tuple[npt.NDArray[np.float64], float] = np.array([2.0, -3.0]), 0.0

    @staticmethod
    def _value(x: npt.NDArray[np.float64]) -> float:
        x1 = float(x[0])
        x2 = float(x[1])

        return (x1 - 2) ** 2 + (x2 + 3) ** 2

    @staticmethod
    def _gradient(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])
        x2 = float(x[1])

        dx1 = 2 * (x1 - 2)
        dx2 = 2 * (x2 + 3)

        return np.array([dx1, dx2])

    @staticmethod
    def _hessian_matrix(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        dx1dx1 = 2
        dx2dx2 = 2
        dx1dx2 = 0
        dx2dx1 = 0

        return np.array([[dx1dx1, dx1dx2], [dx2dx1, dx2dx2]])

    @staticmethod
    def system_of_equations(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])
        x2 = float(x[1])

        return np.array([x1 - 2, x2 + 3])

    @staticmethod
    def jacobian(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[1, 0], [0, 1]])


@dataclasses.dataclass
class Function4(BaseFunction):
    def __post_init__(self: Function1) -> None:
        self.starting_point: npt.NDArray[np.float64] = np.array([0.0, 0.0])
        self.minimum: Tuple[npt.NDArray[np.float64], float] = (
            np.array([-1.769292354238631415240409464, 1.0]), -4.219136248741586519298142210
        )

    @staticmethod
    def _value(x: npt.NDArray[np.float64]) -> float:
        if x.size != 2:
            raise ValueError("only 2D values supported")

        x1 = float(x[0])
        x2 = float(x[1])

        return 0.25 * x1 ** 4 - x1 ** 2 + 2 * x1 + (x2 - 1) ** 2

    @staticmethod
    def _gradient(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])
        x2 = float(x[1])

        dx1 = x1 ** 3 - 2 * x1 + 2
        dx2 = 2 * (x2 - 1)

        return np.array([dx1, dx2])

    @staticmethod
    def _hessian_matrix(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = float(x[0])

        dx1dx1 = 3 * x1 ** 2 - 2
        dx2dx2 = 2
        dx1dx2 = 0
        dx2dx1 = 0

        return np.array([[dx1dx1, dx1dx2], [dx2dx1, dx2dx2]])
