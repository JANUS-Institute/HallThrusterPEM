from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from .types import Array

T = TypeVar("T", np.float64, Array)


@dataclass(frozen=True)
class Measurement(Generic[T]):
    """A measurement object that includes a mean and standard deviation. The mean is the best estimate of the
    quantity being measured, and the standard deviation is the uncertainty in the measurement. Can be used to specify
    a scalar measurement quantity or a field quantity (e.g. a profile) in the form of a `numpy` array.
    """

    mean: T
    std: T

    def __str__(self):
        return f"(μ = {self.mean}, σ = {self.std})"

    @staticmethod
    def gauss_logpdf(data: 'Measurement[T] | None', observation: 'Measurement[T] | None') -> np.float64:
        if data is None or observation is None:
            return np.float64(0.0)

        return _gauss_logpdf(data.mean, data.std, observation.mean)

    @staticmethod
    def interp_gauss_logpdf(
        coords: Array | None,
        data: 'Measurement[Array] | None',
        obs_coords: Array | None,
        observation: 'Measurement[Array] | None',
    ) -> np.float64:
        if coords is None or data is None or obs_coords is None or observation is None:
            return np.float64(0.0)

        obs_interp_mean = np.interp(coords, obs_coords, observation.mean)
        obs_interp = Measurement(obs_interp_mean, np.zeros(1))
        return Measurement.gauss_logpdf(data, obs_interp)


def _gauss_logpdf(mean: T, std: T, observation: T) -> np.float64:
    return -0.5 * np.sum(2 * np.log(std) + (mean - observation) ** 2 / (std**2))
