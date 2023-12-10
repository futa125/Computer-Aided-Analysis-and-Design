import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lab3.functions import Function1, Function2, Function3, Function4, BaseFunction
from lab3.optimisations import gradient_descent, newton_raphson, gauss_newton


def task1() -> None:
    print("# Task 1")

    f: BaseFunction = Function3()

    point, value = gradient_descent(f.starting_point, f, use_golden_ratio=True)
    min_point, min_value = f.minimum

    assert np.allclose(point, min_point)
    assert np.isclose(value, min_value)

    print(point, value)

    try:
        gradient_descent(f.starting_point, f, use_golden_ratio=False)
    except Exception as e:
        print(e)

    print()


def task2() -> None:
    print("# Task 2")

    f1: BaseFunction = Function1()

    point, value = gradient_descent(f1.starting_point, f1)
    min_point, min_value = f1.minimum

    assert np.allclose(point, min_point, atol=10e-5)
    assert np.isclose(value, min_value)

    print(f"Function 1 + Gradient Descent: {point}, {value}, "
          f"calculations: {f1.call_count}, {f1.gradient_calculation_count}, {f1.hessian_matrix_calculation_count}")

    f1: BaseFunction = Function1()

    point, value = newton_raphson(f1.starting_point, f1)
    min_point, min_value = f1.minimum

    assert np.allclose(point, min_point)
    assert np.isclose(value, min_value)

    print(f"Function 1 + Newton-Raphson: {point}, {value}, "
          f"calculations: {f1.call_count}, {f1.gradient_calculation_count}, {f1.hessian_matrix_calculation_count}")

    f2: BaseFunction = Function2()

    point, value = gradient_descent(f2.starting_point, f2)
    min_point, min_value = f2.minimum

    assert np.allclose(point, min_point)
    assert np.isclose(value, min_value)

    print(f"Function 2 + Gradient Descent: {point}, {value}, "
          f"calculations: {f2.call_count}, {f2.gradient_calculation_count}, {f2.hessian_matrix_calculation_count}")

    f2: BaseFunction = Function2()

    point, value = newton_raphson(f2.starting_point, f2)
    min_point, min_value = f2.minimum

    assert np.allclose(point, min_point)
    assert np.isclose(value, min_value)

    print(f"Function 2 + Newton-Raphson: {point}, {value}, "
          f"calculations: {f2.call_count}, {f2.gradient_calculation_count}, {f2.hessian_matrix_calculation_count}")

    print()


def task3() -> None:
    print("# Task 3")

    f: BaseFunction = Function4()

    point, value = newton_raphson(np.array([3.0, 3.0]), f, use_golden_ratio=False)
    min_point, min_value = f.minimum

    assert np.allclose(point, min_point)
    assert np.isclose(value, min_value)

    print(f"Starting point (3, 3): {point}, {value}, "
          f"calculations: {f.call_count}, {f.gradient_calculation_count}, {f.hessian_matrix_calculation_count}")

    f: BaseFunction = Function4()

    try:
        newton_raphson(np.array([1.0, 2.0]), f, use_golden_ratio=False)
    except Exception as e:
        print(f"Starting point (1, 2): {e}")

    f: BaseFunction = Function4()

    point, value = newton_raphson(np.array([3.0, 3.0]), f, use_golden_ratio=True)
    min_point, min_value = f.minimum

    assert np.allclose(point, min_point)
    assert np.isclose(value, min_value)

    print(f"Starting point (3, 3) + Golden Ratio: {point}, {value}, "
          f"calculations: {f.call_count}, {f.gradient_calculation_count}, {f.hessian_matrix_calculation_count}")

    f: BaseFunction = Function4()

    point, value = newton_raphson(np.array([1.0, 2.0]), f, use_golden_ratio=True)
    min_point, min_value = f.minimum

    assert np.allclose(point, min_point)
    assert np.isclose(value, min_value)

    print(f"Starting point (1, 2) + Golden Ratio: {point}, {value}, "
          f"calculations: {f.call_count}, {f.gradient_calculation_count}, {f.hessian_matrix_calculation_count}")

    print()


def task4() -> None:
    print("# Task 4")

    f: BaseFunction = Function1()

    point, value = gauss_newton(f.starting_point, f)
    min_point, min_value = f.minimum

    assert np.allclose(point, min_point)
    assert np.isclose(value, min_value)

    print(f"Gauss-Newton: {point}, {value}, calculations: {f.call_count}")

    print()


def task5() -> None:
    print("# Task 5")

    class Function(BaseFunction):
        @staticmethod
        def _value(x: npt.NDArray[np.float64]) -> float:
            x1 = float(x[0])
            x2 = float(x[1])

            return (x1 ** 2 + x2 ** 2 - 1) ** 2 + (x2 - x1 ** 2) ** 2

        @staticmethod
        def system_of_equations(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            x1 = float(x[0])
            x2 = float(x[1])

            return np.array([x1 ** 2 + x2 ** 2 - 1, x2 - x1 ** 2])

        @staticmethod
        def jacobian(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            x1 = float(x[0])
            x2 = float(x[1])

            return np.array([[2 * x1, 2 * x2], [-2 * x1, 1]])

    f: BaseFunction = Function()
    point, value = gauss_newton(np.array([-2.0, 2.0]), f)
    print(f"Gauss-Newton at (-2, 2): {point}, {value}, calculations: {f.call_count}")

    f: BaseFunction = Function()
    point, value = gauss_newton(np.array([2.0, 2.0]), f)
    print(f"Gauss-Newton at (2, 2): {point}, {value}, calculations: {f.call_count}")

    f: BaseFunction = Function()
    point, value = gauss_newton(np.array([2.0, -2.0]), f)
    print(f"Gauss-Newton at (2, -2): {point}, {value}, calculations: {f.call_count}")

    print()


def task6() -> None:
    print("# Task 6")

    class Function(BaseFunction):
        @staticmethod
        def _value(x: npt.NDArray[np.float64]) -> float:
            x1 = float(x[0])
            x2 = float(x[1])
            x3 = float(x[2])

            return ((x1 * math.exp(1 * x2) + x3 - 3) ** 2 + (x1 * math.exp(2 * x2) + x3 - 4) ** 2 +
                    (x1 * math.exp(3 * x2) + x3 - 4) ** 2 + (x1 * math.exp(5 * x2) + x3 - 5) ** 2 +
                    (x1 * math.exp(6 * x2) + x3 - 6) ** 2 + (x1 * math.exp(7 * x2) + x3 - 8) ** 2)

        @staticmethod
        def system_of_equations(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            x1 = float(x[0])
            x2 = float(x[1])
            x3 = float(x[2])

            return np.array([
                x1 * math.exp(1 * x2) + x3 - 3,
                x1 * math.exp(2 * x2) + x3 - 4,
                x1 * math.exp(3 * x2) + x3 - 4,
                x1 * math.exp(5 * x2) + x3 - 5,
                x1 * math.exp(6 * x2) + x3 - 6,
                x1 * math.exp(7 * x2) + x3 - 8,
            ])

        @staticmethod
        def jacobian(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            x1 = float(x[0])
            x2 = float(x[1])
            x3 = float(x[2])

            return np.array([
                [math.exp(1 * x2), 1 * x1 * math.exp(1 * x2), 1],
                [math.exp(2 * x2), 2 * x1 * math.exp(2 * x2), 1],
                [math.exp(3 * x2), 3 * x1 * math.exp(3 * x2), 1],
                [math.exp(5 * x2), 5 * x1 * math.exp(5 * x2), 1],
                [math.exp(6 * x2), 6 * x1 * math.exp(6 * x2), 1],
                [math.exp(7 * x2), 7 * x1 * math.exp(7 * x2), 1],
            ])

    f: BaseFunction = Function()
    point, value = gauss_newton(np.array([1.0, 1.0, 1.0]), f, use_golden_ratio=True)
    print(f"Gauss-Newton at (1, 1, 1): {point}, {value}, calculations: {f.call_count}")

    t_values = np.linspace(0, 10)
    y_values = point[0] * np.exp(point[1] * t_values) + point[2]

    plt.plot(t_values, y_values, label="M(x, t)")
    plt.scatter(np.array([1, 2, 3, 5, 6, 7]), np.array([3, 4, 4, 5, 6, 8]), color="red")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("M(x, t)")
    plt.show()

    print()


def main() -> None:
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()


if __name__ == "__main__":
    main()
