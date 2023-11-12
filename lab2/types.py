from typing import Tuple, Callable

import numpy as np
from numpy.typing import NDArray

Point = NDArray[np.float64]
Interval = Tuple[Point, Point]
ObjectiveFunction = Callable[[Point], float]
