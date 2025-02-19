from abc import ABC
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.floating[Any]]
PathLike: TypeAlias = str | Path


class ThrusterDataset(ABC):
    pass
