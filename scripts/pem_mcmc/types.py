from typing import TypeAlias

import numpy as np

from hallmd.data import Array, OperatingCondition, ThrusterData

Number: TypeAlias = np.float64
Value: TypeAlias = Number | Array
NaN = np.float64(np.nan)
Inf = np.float64(np.inf)

Dataset: TypeAlias = dict[OperatingCondition, ThrusterData]
