from lab2.types import Point, ObjectiveFunction


def hooke_jevees(starting_point: Point, f: ObjectiveFunction, dx=0.5, e=10e-6, logging_enabled: bool = False):
    xp = starting_point.copy()
    xb = starting_point.copy()

    while True:
        xn = search(xp, f, dx)

        if logging_enabled:
            print(f"xp={xp}, xb={xb}, xn={xn}")
            print(f"f(xp)={f(xp)}, f(xb)={f(xb)}, f(xn)={f(xn)}")

        if f(xn) < f(xb):
            for i in range(len(xp)):
                xp[i] = 2 * xn[i] - xb[i]

            xb = xn.copy()

        else:
            dx *= 0.5
            xp = xb.copy()

        if dx <= 0.5*e:
            return xb


def search(xp: Point, f: ObjectiveFunction, dx: float) -> Point:
    x = xp.copy()

    for i in range(len(x)):
        p = f(x)

        x[i] += dx
        n = f(x)

        if n > p:
            x[i] -= 2 * dx
            n = f(x)

            if n > p:
                x[i] += dx

    return x
