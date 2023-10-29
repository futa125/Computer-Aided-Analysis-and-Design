from __future__ import annotations

import dataclasses
import math
from copy import deepcopy
from typing import List, Optional, Tuple

EPSILON = 10e-9


def set_epsilon(value: float) -> None:
    global EPSILON
    EPSILON = value


@dataclasses.dataclass
class Matrix:
    row_count: Optional[int] = dataclasses.field(init=True, default=None)
    column_count: Optional[int] = dataclasses.field(init=True, default=None)
    filename: Optional[str] = dataclasses.field(init=True, default=None)

    matrix_a: Matrix = dataclasses.field(init=False)
    matrix_p: Matrix = dataclasses.field(init=False)

    row_swap_count: int = dataclasses.field(init=False, default=0)

    elements: List[List[float]] = dataclasses.field(init=False, default_factory=list)

    def __post_init__(self: Matrix) -> None:
        if self.filename is not None:
            self._load_from_file()
            return

        if self.row_count is not None and self.column_count is not None:
            self._init_empty_matrix()
            return

        self.row_count, self.column_count = 0, 0
        self._init_empty_matrix()

    def __getitem__(self: Matrix, indexes: Tuple[int, int]) -> float:
        i, j = indexes

        if i < 0 or i >= self.row_count:
            raise ValueError("row index out of range")

        if j < 0 or j >= self.column_count:
            raise ValueError("column index ouf of range")

        return self.elements[i][j]

    def __setitem__(self: Matrix, indexes: Tuple[int, int], value: float) -> None:
        i, j = indexes

        if i < 0 or i >= self.row_count:
            raise ValueError("row index out of range")

        if j < 0 or j >= self.column_count:
            raise ValueError("column index ouf of range")

        self.elements[i][j] = value

    def __add__(self: Matrix, other: Matrix) -> Matrix:
        if not self._is_add_sub_compatible(other):
            raise ValueError("only matrices with equivalent dimensions can be added together")

        output = Matrix(row_count=self.row_count, column_count=self.column_count)
        for i in range(self.row_count):
            for j in range(self.column_count):
                output[i, j] = self[i, j] + other[i, j]

        return output

    def __iadd__(self: Matrix, other: Matrix) -> Matrix:
        if not self._is_add_sub_compatible(other):
            raise ValueError("only matrices with equivalent dimensions can be added together")

        for i in range(self.row_count):
            for j in range(self.column_count):
                self[i, j] += other[i, j]

        return self

    def __sub__(self: Matrix, other: Matrix) -> Matrix:
        if not self._is_add_sub_compatible(other):
            raise ValueError("only matrices with equivalent dimensions can be subtracted")

        output = Matrix(row_count=self.row_count, column_count=self.column_count)
        for i in range(self.row_count):
            for j in range(self.column_count):
                output[i, j] = self[i, j] - other[i, j]

        return output

    def __isub__(self: Matrix, other: Matrix) -> Matrix:
        if not self._is_add_sub_compatible(other):
            raise ValueError("only matrices with equivalent dimensions can be subtracted")

        for i in range(self.row_count):
            for j in range(self.column_count):
                self[i, j] -= other[i, j]

        return self

    def __mul__(self: Matrix, other: int | float | Matrix) -> Matrix:
        # Matrix multiplication
        if isinstance(self, Matrix) and isinstance(other, Matrix):
            if not self._is_mul_compatible(other):
                raise ValueError(f"incompatible matrix dimensions")

            output = Matrix(row_count=self.row_count, column_count=other.column_count)
            for i in range(self.row_count):
                for j in range(other.column_count):
                    for k in range(other.row_count):
                        output[i, j] += self[i, k] * other[k, j]

            return output

        # Scalar multiplication
        if isinstance(self, Matrix) and (isinstance(other, int) or isinstance(other, float)):
            output = Matrix(row_count=self.row_count, column_count=self.column_count)
            for i in range(self.row_count):
                for j in range(self.column_count):
                    output[i, j] = self[i, j] * other

            return output

        raise ValueError("matrices can only be multiplied with each other or with scaler values")

    def __rmul__(self: Matrix, other: int | float | Matrix) -> Matrix:
        return self.__mul__(other)

    def __eq__(self: Matrix, other: Matrix) -> bool:
        if not isinstance(self, Matrix) or not isinstance(other, Matrix):
            raise ValueError("only matrices can be checked for equality")

        if self.row_count != other.row_count:
            return False

        if self.column_count != other.column_count:
            return False

        return all([self[i, j] == other[i, j] for j in range(self.column_count) for i in range(self.row_count)])

    def __str__(self: Matrix):
        out: str = ""
        for row in self.elements:
            out += " ".join(str(x) for x in row)
            out += "\n"

        return out

    def __invert__(self: Matrix) -> Matrix:
        out = Matrix(row_count=self.column_count, column_count=self.row_count)
        for i in range(self.row_count):
            for j in range(self.column_count):
                out[j, i] = self[i, j]

        return out

    def _init_empty_matrix(self: Matrix) -> None:
        self.elements = [[0.0 for _ in range(self.column_count)] for _ in range(self.row_count)]

    def _load_from_file(self: Matrix) -> None:
        with open(self.filename, "r") as file:
            lines = file.readlines()

        self.row_count = len(lines)

        i: int
        line: str
        for i, line in enumerate(lines):
            row_elements = line.strip().split(" ")

            if i == 0:
                self.column_count = len(row_elements)
                self._init_empty_matrix()

            j: int
            element: str
            for j, element in enumerate(row_elements):
                fraction = element.split("/")
                if len(fraction) == 2:
                    a, b = fraction
                    self[i, j] = float(a) / float(b)

                    continue

                self[i, j] = float(element)

    def _is_add_sub_compatible(self: Matrix, other: Matrix) -> bool:
        if not isinstance(self, Matrix) or not isinstance(other, Matrix):
            raise ValueError("both values have to be matrices")

        return self.row_count == other.row_count and self.column_count == other.column_count

    def _is_mul_compatible(self: Matrix, other: Matrix) -> bool:
        if not isinstance(self, Matrix) or not isinstance(other, Matrix):
            raise ValueError("both values have to be matrices")

        return self.column_count == other.row_count

    def _is_square(self: Matrix) -> bool:
        return self.row_count == self.column_count

    def _is_lower_triangular(self: Matrix) -> bool:
        for i in range(self.row_count):
            for j in range(i + 1, self.column_count):
                if self[i, j] != 0:
                    return False

        return True

    def _is_upper_triangular(self: Matrix) -> bool:
        for i in range(self.row_count):
            for j in range(i):
                if self[i, j] != 0:
                    return False

        return True

    def _swap_rows(self: Matrix, x: int, y: int) -> None:
        tmp = self.elements[x]
        self.elements[x] = self.elements[y]
        self.elements[y] = tmp

    def _extract_column(self: Matrix, j: int) -> Matrix:
        out = Matrix(row_count=self.row_count, column_count=1)
        for i in range(self.row_count):
            out[i, 0] = self[i, j]

        return out

    def _decomposition_step(self: Matrix, i: int) -> None:
        for j in range(i + 1, self.column_count):
            if abs(self.matrix_a[i, i]) < EPSILON:
                raise ValueError("matrix is singular")

            self.matrix_a[j, i] = self.matrix_a[j, i] / self.matrix_a[i, i]
            for k in range(i + 1, self.row_count):
                self.matrix_a[j, k] = self.matrix_a[j, k] - self.matrix_a[j, i] * self.matrix_a[i, k]

    def _forward_substitution(self: Matrix, b: Matrix):
        if not self._is_square():
            raise ValueError("L needs to be a square matrix")

        if self.row_count != b.row_count:
            raise ValueError("L and b need to have the same number of rows")

        if b.column_count != 1:
            raise ValueError("b is not a vector")

        new_matrix = deepcopy(b)
        for i in range(self.row_count - 1):
            for j in range(i + 1, self.column_count):
                new_matrix[j, 0] -= self.matrix_a[j, i] * new_matrix[i, 0]

        return new_matrix

    def _back_substitution(self: Matrix, b: Matrix):
        if not self._is_square():
            raise ValueError("U needs to be a square matrix")

        if self.row_count != b.row_count:
            raise ValueError("U and b need to have the same number of rows")

        if b.column_count != 1:
            raise ValueError("b is not a vector")

        new_matrix = deepcopy(b)
        for i in range(self.row_count - 1, -1, -1):
            if abs(self.matrix_a[i, i]) < EPSILON:
                raise ValueError("input is a singular matrix")

            new_matrix[i, 0] /= self.matrix_a[i, i]
            for j in range(i):
                new_matrix[j, 0] -= self.matrix_a[j, i] * new_matrix[i, 0]

        return new_matrix

    def save_to_file(self: Matrix, filename: str) -> None:
        with open(filename, "w") as file:
            file.writelines(self.__str__())

    def lu_decomposition(self: Matrix) -> None:
        if not self._is_square():
            raise ValueError("input needs to be a square matrix")

        self.matrix_a = deepcopy(self)

        for i in range(0, self.row_count):
            self._decomposition_step(i)

    def lup_decomposition(self: Matrix):
        if not self._is_square():
            raise ValueError("input needs to be a square matrix")

        self.matrix_a = deepcopy(self)
        self.matrix_p = Matrix(row_count=self.row_count, column_count=self.column_count)
        for i in range(self.row_count):
            self.matrix_p[i, i] = 1

        self.row_swap_count = 0

        for i in range(0, self.row_count):
            pivot_value = 0
            pivot = i

            for j in range(i, self.row_count):
                if abs(self.matrix_a[j, i]) > pivot_value:
                    pivot_value = abs(self.matrix_a[j, i])
                    pivot = j

            if pivot != i:
                self.matrix_a._swap_rows(pivot, i)
                self.matrix_p._swap_rows(pivot, i)
                self.row_swap_count += 1

            self._decomposition_step(i)

    def inverse(self: Matrix) -> Optional[Matrix]:
        try:
            self.lup_decomposition()

            out = Matrix(row_count=self.row_count, column_count=self.column_count)
            for i in range(self.row_count):
                e = self.matrix_p._extract_column(i)
                y = self._forward_substitution(e)
                x = self._back_substitution(y)

                for j in range(x.row_count):
                    out[j, i] = x[j, 0]

            return out

        except ValueError:
            return None

    def determinant(self: Matrix) -> float:
        try:
            self.lup_decomposition()
            value = (-1)**self.row_swap_count
            value *= math.prod(self.matrix_a[i, i] for i in range(self.matrix_a.row_count))

            return value

        except ValueError:
            return 0

    def lu_solve(self: Matrix, b: Matrix) -> Optional[Matrix]:
        try:
            self.lu_decomposition()
            y = self._forward_substitution(b)
            x = self._back_substitution(y)

        except ValueError:
            return None

        return x

    def lup_solve(self: Matrix, b: Matrix) -> Optional[Matrix]:
        try:
            self.lup_decomposition()
            y = self._forward_substitution(self.matrix_p * b)
            x = self._back_substitution(y)

        except ValueError:
            return None

        return x
