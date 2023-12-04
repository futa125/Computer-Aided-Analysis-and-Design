from lab3.functions import Function1, Function2, Function3
from lab3.optimisations.gradient import gradient_descent


def main() -> None:
    f1 = Function3()
    minimum = gradient_descent(f1.starting_point(), f1)
    print(minimum, f1.minimum())


if __name__ == "__main__":
    main()
