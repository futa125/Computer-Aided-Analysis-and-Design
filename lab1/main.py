import math
from typing import List

from lab1.matrix import Matrix, set_epsilon, EPSILON


def assert_matrix_elements_nearly_equal(a: List[List[float]], b: List[List[float]]) -> None:
    if len(a) != len(b):
        raise AssertionError("matrices must have equal row counts")

    x: List[float]
    y: List[float]
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            raise AssertionError("matrices must have equal column counts")

        for value_a, value_b in zip(row_a, row_b):
            tol = 1e-2
            assert math.isclose(value_a, value_b, abs_tol=tol), f"abs({value_a} - {value_b}) > {tol}"


def print_delimiter() -> None:
    print("-" * 40)


def main() -> None:
    # Task 01
    print("# Task 01")
    a = Matrix(filename="tasks/task01/A.txt")
    b = Matrix(filename="tasks/task01/A.txt")

    print(f"Matrix A before operations:\n{a}")
    assert a == b

    a *= (1.0 + 0.1 + 0.1 + 0.1)
    a *= 1/1.3

    assert a != b

    print(f"Matrix A after operations:\n{a}")

    assert_matrix_elements_nearly_equal(a.elements, b.elements)

    print_delimiter()

    # Task 02
    print("# Task 02")
    matrix = Matrix(filename="tasks/task02/A.txt")
    b = Matrix(filename="tasks/task02/b.txt")

    lu_solution = matrix.lu_solve(b)
    assert lu_solution is None

    lup_solution = matrix.lup_solve(b)
    assert_matrix_elements_nearly_equal(
        lup_solution.elements, [
            [3],
            [1],
            [-1],
        ]
    )

    print(f"LU Solution:\n{lu_solution}\n")
    print(f"LUP Solution:\n{lup_solution}")

    with open("tasks/task02/lu_solution.txt", "w") as file:
        file.write(f"{lu_solution}\n")

    lup_solution.save_to_file("tasks/task02/lup_solution.txt")

    print_delimiter()

    # Task 03
    print("# Task 03")
    matrix = Matrix(filename="tasks/task03/A.txt")
    b = Matrix(filename="tasks/task03/b.txt")

    lu_solution = matrix.lu_solve(b)
    assert lu_solution is None

    lup_solution = matrix.lup_solve(b)
    assert lup_solution is None

    print(f"LU Solution:\n{lu_solution}\n")
    print(f"LUP Solution:\n{lup_solution}\n")

    with open("tasks/task02/lu_solution.txt", "w") as file:
        file.write(f"{lu_solution}\n")

    with open("tasks/task02/lup_solution.txt", "w") as file:
        file.write(f"{lup_solution}\n")

    print_delimiter()

    # Task 04
    print("# Task 04")
    matrix = Matrix(filename="tasks/task04/A.txt")
    b = Matrix(filename="tasks/task04/b.txt")

    lu_solution = matrix.lu_solve(b)
    assert_matrix_elements_nearly_equal(
        lu_solution.elements, [
            [1],
            [2],
            [3],
        ],
    )

    lup_solution = matrix.lup_solve(b)
    assert_matrix_elements_nearly_equal(
        lup_solution.elements, [
            [1],
            [2],
            [3],
        ],
    )

    print(f"LU Solution:\n{lu_solution}")
    print(f"LUP Solution:\n{lup_solution}")

    lu_solution.save_to_file("tasks/task04/lu_solution.txt")
    lup_solution.save_to_file("tasks/task04/lup_solution.txt")

    print_delimiter()

    # Task 05
    print("# Task 05")
    matrix = Matrix(filename="tasks/task05/A.txt")
    b = Matrix(filename="tasks/task05/b.txt")

    lu_solution = matrix.lu_solve(b)
    assert lu_solution is None

    lup_solution = matrix.lup_solve(b)
    assert_matrix_elements_nearly_equal(
        lup_solution.elements, [
            [0],
            [0],
            [3],
        ],
    )

    print(f"LU Solution:\n{lu_solution}\n")
    print(f"LUP Solution:\n{lup_solution}")

    with open("tasks/task05/lu_solution.txt", "w") as file:
        file.write(f"{lu_solution}\n")

    lup_solution.save_to_file("tasks/task05/lup_solution.txt")

    print_delimiter()

    # Task 06
    print("# Task 06")
    epsilon = EPSILON
    set_epsilon(10e-6)

    matrix = Matrix(filename="tasks/task06/A.txt")
    b = Matrix(filename="tasks/task06/b.txt")

    lu_solution = matrix.lu_solve(b)
    assert lu_solution is None

    lup_solution = matrix.lup_solve(b)
    assert lup_solution is None

    print(f"LU Solution:\n{lu_solution}\n")
    print(f"LUP Solution:\n{lup_solution}\n")

    with open("tasks/task06/lu_solution.txt", "w") as file:
        file.write(f"{lu_solution}\n")

    with open("tasks/task06/lup_solution.txt", "w") as file:
        file.write(f"{lup_solution}\n")

    set_epsilon(epsilon)
    print_delimiter()

    # Task 07
    print("# Task 07")
    matrix = Matrix(filename="tasks/task07/A.txt")
    inv = matrix.inverse()

    assert inv is None

    print(f"Inverse:\n{inv}\n")
    with open("tasks/task07/inverse.txt", "w") as file:
        file.write(f"{inv}\n")

    print_delimiter()

    # Task 08
    print("# Task 08")
    matrix = Matrix(filename="tasks/task08/A.txt")
    inv = matrix.inverse()

    assert_matrix_elements_nearly_equal(
        inv.elements, [
            [0, -3, -2],
            [1, -4, -2],
            [-3, 4, 1],
        ],
    )

    print(f"Inverse:\n{inv}")

    inv.save_to_file("tasks/task08/inverse.txt")

    print_delimiter()

    # Task 09
    print("# Task 09")
    matrix = Matrix(filename="tasks/task09/A.txt")
    det = matrix.determinant()

    assert det == 1

    print(f"Determinant:\n{det}\n")
    with open("tasks/task09/determinant.txt", "w") as file:
        file.write(f"{det}\n")

    print_delimiter()

    # Task 10
    print("# Task 10")
    matrix = Matrix(filename="tasks/task10/A.txt")
    det = matrix.determinant()

    assert det == 48

    print(f"Determinant:\n{det}\n")
    with open("tasks/task10/determinant.txt", "w") as file:
        file.write(f"{det}\n")


if __name__ == "__main__":
    main()
