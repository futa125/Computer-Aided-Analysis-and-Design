import math
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from lab5.corrector import reverse_euler_corrector, trapezoidal_corrector
from lab5.methods import euler, predictor_corrector, reverse_euler, runge_kutta, trapezoidal
from lab5.predictor import euler_predictor


def cumulative_error(
        predicted: List[npt.NDArray[np.float64]],
        actual: List[npt.NDArray[np.float64]],
) -> Tuple[float, float]:
    x1_error: float = 0.0
    x2_error: float = 0.0

    for (x1_predicted, x2_predicted), (x1_actual, x2_actual) in zip(predicted, actual):
        x1_error += abs((x1_predicted[0]) - (x1_actual[0]))
        x2_error += abs((x2_predicted[0]) - (x2_actual[0]))

    return x1_error, x2_error


def task_1() -> None:
    print("# Task 1")

    a: npt.NDArray[np.float64] = np.array([
        [0, 1],
        [-1, 0],
    ])

    b: npt.NDArray[np.float64] = np.array([
        [0, 0],
        [0, 0],
    ])

    starting_state: npt.NDArray[np.float64] = np.array([
        [1],
        [1],
    ])

    t_start = 0
    t_max = 10
    t_step = 0.01

    x_analytic: List[npt.NDArray[np.float64]] = []
    values = np.linspace(t_start, t_max, int((t_max - t_start) / t_step))
    for t in values:
        x_analytic.append(np.array([
            [math.cos(t) + math.sin(t)],
            [math.cos(t) - math.sin(t)],
        ]))

    x_runge_kutta: List[npt.NDArray[np.float64]] = runge_kutta(a, b, starting_state, t_start, t_max, t_step)
    x_trapezoidal: List[npt.NDArray[np.float64]] = trapezoidal(a, b, starting_state, t_start, t_max, t_step)
    x_euler: List[npt.NDArray[np.float64]] = euler(a, b, starting_state, t_start, t_max, t_step)
    x_reverse_euler: List[npt.NDArray[np.float64]] = reverse_euler(a, b, starting_state, t_start, t_max, t_step)
    x_pece_2: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, t_step,
                                                                  euler_predictor, reverse_euler_corrector, 2)
    x_pece: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, t_step,
                                                                euler_predictor, trapezoidal_corrector, 1)

    print("Runge kutta: ", cumulative_error(x_runge_kutta, x_analytic))
    print("Trapezoidal: ", cumulative_error(x_trapezoidal, x_analytic))
    print("Euler: ", cumulative_error(x_euler, x_analytic))
    print("Reverse euler: ", cumulative_error(x_reverse_euler, x_analytic))
    print("PE(CE)²: ", cumulative_error(x_pece_2, x_analytic))
    print("PECE: ", cumulative_error(x_pece, x_analytic))

    data = [x_analytic, x_runge_kutta, x_trapezoidal, x_euler, x_reverse_euler, x_pece_2, x_pece]
    labels = ["Analytic", "Runge-Kutta", "Trapezoidal", "Euler", "Reverse-Euler", "PE(CE)²", "PECE"]

    t = np.linspace(t_start, t_max, int((t_max - t_start) / t_step))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()
    ax1.set_title("x1")
    ax2.set_title("x2")

    for label, x in zip(labels, data):
        x = np.hstack(x)
        ax1.plot(t, x[0], label=label)

    for label, x in zip(labels, data):
        x = np.hstack(x)
        ax2.plot(t, x[1], label=label)

    ax1.legend()
    ax2.legend()
    plt.show()

    print()


def task_2() -> None:
    print("# Task 2")

    a: npt.NDArray[np.float64] = np.array([
        [0, 1],
        [-200, -102],
    ])

    b: npt.NDArray[np.float64] = np.array([
        [0, 0],
        [0, 0],
    ])

    starting_state: npt.NDArray[np.float64] = np.array([
        [1],
        [-2],
    ])

    t_start = 0
    t_max = 1
    t_step = 0.1

    x_analytic: List[npt.NDArray[np.float64]] = []
    values = np.linspace(t_start, t_max, int((t_max - t_start) / t_step))
    for t in values:
        x_analytic.append(np.array([
            [1 * math.cos(t) - 2 * math.sin(t)],
            [1 * math.cos(t) + 2 * math.sin(t)],
        ]))

    x_runge_kutta: List[npt.NDArray[np.float64]] = runge_kutta(a, b, starting_state, t_start, t_max, t_step)
    x_trapezoidal: List[npt.NDArray[np.float64]] = trapezoidal(a, b, starting_state, t_start, t_max, t_step)
    x_euler: List[npt.NDArray[np.float64]] = euler(a, b, starting_state, t_start, t_max, t_step)
    x_reverse_euler: List[npt.NDArray[np.float64]] = reverse_euler(a, b, starting_state, t_start, t_max, t_step)
    x_pece_2: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, t_step,
                                                                  euler_predictor, reverse_euler_corrector, 2)
    x_pece: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, t_step,
                                                                euler_predictor, trapezoidal_corrector, 1)

    print("Runge kutta: ", cumulative_error(x_runge_kutta, x_analytic))
    print("Trapezoidal: ", cumulative_error(x_trapezoidal, x_analytic))
    print("Euler: ", cumulative_error(x_euler, x_analytic))
    print("Reverse euler: ", cumulative_error(x_reverse_euler, x_analytic))
    print("PE(CE)²: ", cumulative_error(x_pece_2, x_analytic))
    print("PECE: ", cumulative_error(x_pece, x_analytic))

    t = np.linspace(t_start, t_max, int((t_max - t_start) / t_step))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()

    x = np.hstack(x_runge_kutta)
    ax1.plot(t, x[0], label="x1")
    ax1.plot(t, x[1], label="x2")
    ax1.set_title("Runge-Kutta")

    x = np.hstack(x_pece_2)
    ax2.plot(t, x[0], label="x1")
    ax2.plot(t, x[1], label="x2")
    ax2.set_title("PE(CE)²")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.show()

    x_runge_kutta: List[npt.NDArray[np.float64]] = runge_kutta(a, b, starting_state, t_start, t_max, 0.01)
    x_pece_2: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, 0.01,
                                                                  euler_predictor, reverse_euler_corrector, 2)

    print("Runge kutta with step 0.01: ", cumulative_error(x_runge_kutta, x_analytic))
    print("PE(CE)² with step 0.01: ", cumulative_error(x_pece_2, x_analytic))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()

    t = np.linspace(t_start, t_max, int((t_max - t_start) / 0.01))

    x = np.hstack(x_runge_kutta)
    ax1.plot(t, x[0], label="x1")
    ax1.plot(t, x[1], label="x2")
    ax1.set_title("Runge-Kutta")

    x = np.hstack(x_pece_2)
    ax2.plot(t, x[0], label="x1")
    ax2.plot(t, x[1], label="x2")
    ax2.set_title("PE(CE)²")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.show()

    print()


def task_3() -> None:
    print("# Task 3")
    print("plotting...")

    a: npt.NDArray[np.float64] = np.array([
        [0, -2],
        [1, -3],
    ])

    b: npt.NDArray[np.float64] = np.array([
        [2, 0],
        [0, 3],
    ])

    starting_state: npt.NDArray[np.float64] = np.array([
        [1],
        [3],
    ])

    def r(_: float) -> npt.NDArray[np.float64]:
        return np.array([
            [1],
            [1],
        ])

    t_start = 0
    t_max = 10
    t_step = 0.01

    x_runge_kutta: List[npt.NDArray[np.float64]] = runge_kutta(a, b, starting_state, t_start, t_max, t_step, r)
    x_trapezoidal: List[npt.NDArray[np.float64]] = trapezoidal(a, b, starting_state, t_start, t_max, t_step, r)
    x_euler: List[npt.NDArray[np.float64]] = euler(a, b, starting_state, t_start, t_max, t_step, r)
    x_reverse_euler: List[npt.NDArray[np.float64]] = reverse_euler(a, b, starting_state, t_start, t_max, t_step, r)
    x_pece_2: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, t_step,
                                                                  euler_predictor, reverse_euler_corrector, 2, r)
    x_pece: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, t_step,
                                                                euler_predictor, trapezoidal_corrector, 1, r)

    data = [x_runge_kutta, x_trapezoidal, x_euler, x_reverse_euler, x_pece_2, x_pece]
    labels = ["Runge-Kutta", "Trapezoidal", "Euler", "Reverse-Euler", "PE(CE)²", "PECE"]

    t = np.linspace(t_start, t_max, int((t_max - t_start) / t_step))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()
    ax1.set_title("x1")
    ax2.set_title("x2")

    for label, x in zip(labels, data):
        x = np.hstack(x)
        ax1.plot(t, x[0], label=label)

    for label, x in zip(labels, data):
        x = np.hstack(x)
        ax2.plot(t, x[1], label=label)

    ax1.legend()
    ax2.legend()
    plt.show()
    print()


def task_4() -> None:
    print("# Task 4")
    print("plotting...")

    a: npt.NDArray[np.float64] = np.array([
        [1, -5],
        [1, -7],
    ])

    b: npt.NDArray[np.float64] = np.array([
        [5, 0],
        [0, 3],
    ])

    starting_state: npt.NDArray[np.float64] = np.array([
        [-1],
        [3],
    ])

    def r(t_value: float) -> npt.NDArray[np.float64]:
        return np.array([
            [t_value],
            [t_value],
        ])

    t_start = 0
    t_max = 1
    t_step = 0.01

    x_runge_kutta: List[npt.NDArray[np.float64]] = runge_kutta(a, b, starting_state, t_start, t_max, t_step, r)
    x_trapezoidal: List[npt.NDArray[np.float64]] = trapezoidal(a, b, starting_state, t_start, t_max, t_step, r)
    x_euler: List[npt.NDArray[np.float64]] = euler(a, b, starting_state, t_start, t_max, t_step, r)
    x_reverse_euler: List[npt.NDArray[np.float64]] = reverse_euler(a, b, starting_state, t_start, t_max, t_step, r)
    x_pece_2: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, t_step,
                                                                  euler_predictor, reverse_euler_corrector, 2, r)
    x_pece: List[npt.NDArray[np.float64]] = predictor_corrector(a, b, starting_state, t_start, t_max, t_step,
                                                                euler_predictor, trapezoidal_corrector, 1, r)

    data = [x_runge_kutta, x_trapezoidal, x_euler, x_reverse_euler, x_pece_2, x_pece]
    labels = ["Runge-Kutta", "Trapezoidal", "Euler", "Reverse-Euler", "PE(CE)²", "PECE"]

    t = np.linspace(t_start, t_max, int((t_max - t_start) / t_step))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()
    ax1.set_title("x1")
    ax2.set_title("x2")

    for label, x in zip(labels, data):
        x = np.hstack(x)
        ax1.plot(t, x[0], label=label)

    for label, x in zip(labels, data):
        x = np.hstack(x)
        ax2.plot(t, x[1], label=label)

    ax1.legend()
    ax2.legend()
    plt.show()
    print()


def main() -> None:
    task_1()
    task_2()
    task_3()
    task_4()


if __name__ == "__main__":
    main()
