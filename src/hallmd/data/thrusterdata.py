from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np

from .measurement import Measurement
from .types import Array


class ThrusterDataset(ABC):
    """Abstract base class for thruster datasets"""

    @staticmethod
    @abstractmethod
    def datasets_from_names(dataset_names: list[str]) -> list[Path]:
        pass

    @staticmethod
    @abstractmethod
    def all_data() -> list[Path]:
        pass


@dataclass
class CurrentDensitySweep:
    """Contains data for a single current density sweep"""

    radius_m: np.float64
    angles_rad: Array
    current_density_A_m2: Measurement[Array]


@dataclass
class IonVelocityData:
    """Contains measurements of axial ion velocity along with coordinates"""

    axial_distance_m: Array
    velocity_m_s: Measurement[Array]


@dataclass
class ThrusterData:
    """Class for Hall thruster data. Contains fields for all relevant performance metrics and quantities of interest."""

    # Cathode
    cathode_coupling_voltage_V: Optional[Measurement[np.float64]] = None
    # Thruster
    thrust_N: Optional[Measurement[np.float64]] = None
    discharge_current_A: Optional[Measurement[np.float64]] = None
    ion_current_A: Optional[Measurement[np.float64]] = None
    efficiency_current: Optional[Measurement[np.float64]] = None
    efficiency_mass: Optional[Measurement[np.float64]] = None
    efficiency_voltage: Optional[Measurement[np.float64]] = None
    efficiency_anode: Optional[Measurement[np.float64]] = None
    ion_velocity: Optional[IonVelocityData] = None

    # Plume
    ion_current_sweeps: Optional[list[CurrentDensitySweep]] = None

    def __str__(self) -> str:
        fields_str = ",\n".join(
            [
                f"\t{field.name} = {val}"
                for field in fields(ThrusterData)
                if (val := getattr(self, field.name)) is not None
            ]
        )
        return f"ThrusterData(\n{fields_str}\n)\n"

    @staticmethod
    def merge_field(field, data1, data2):
        val1 = getattr(data1, field)
        val2 = getattr(data2, field)
        if val2 is None and val1 is None:
            return None
        elif val2 is None:
            return val1
        else:
            return val2

    @staticmethod
    def update(data1, data2):
        merged = {}
        for field in fields(ThrusterData):
            merged[field.name] = ThrusterData.merge_field(field.name, data1, data2)
        return ThrusterData(**merged)

    @staticmethod
    def log_likelihood(data: 'ThrusterData', observation: 'ThrusterData') -> np.float64:
        log_likelihood = (
            # Add contributions from global performance metrics
            Measurement.gauss_logpdf(data.cathode_coupling_voltage_V, observation.cathode_coupling_voltage_V)
            + Measurement.gauss_logpdf(data.thrust_N, observation.thrust_N)
            + Measurement.gauss_logpdf(data.discharge_current_A, observation.discharge_current_A)
            + Measurement.gauss_logpdf(data.ion_current_A, observation.ion_current_A)
            + Measurement.gauss_logpdf(data.efficiency_current, observation.efficiency_current)
            + Measurement.gauss_logpdf(data.efficiency_mass, observation.efficiency_mass)
            + Measurement.gauss_logpdf(data.efficiency_voltage, observation.efficiency_voltage)
            + Measurement.gauss_logpdf(data.efficiency_anode, observation.efficiency_anode)
        )

        # Contribution to likelihood from ion velocity
        if data.ion_velocity is not None and observation.ion_velocity is not None:
            log_likelihood += Measurement.interp_gauss_logpdf(
                data.ion_velocity.axial_distance_m,
                data.ion_velocity.velocity_m_s,
                observation.ion_velocity.axial_distance_m,
                observation.ion_velocity.velocity_m_s,
            )

        # Contribution to likelihood from ion current density
        plume_log_likelihood = 0
        num_sweeps = 0
        if data.ion_current_sweeps is not None and observation.ion_current_sweeps is not None:
            for data_sweep in data.ion_current_sweeps:
                # find sweep in observation with the same radius, if present
                observation_sweep = None
                for sweep in observation.ion_current_sweeps:
                    if data_sweep.radius_m == sweep.radius_m:
                        observation_sweep = sweep
                        break
                # if no sweep in observation with same radius, do nothing
                if observation_sweep is None:
                    continue

                plume_log_likelihood += Measurement.interp_gauss_logpdf(
                    data_sweep.angles_rad,
                    data_sweep.current_density_A_m2,
                    observation_sweep.angles_rad,
                    observation_sweep.current_density_A_m2,
                )
                num_sweeps += 1

            plume_log_likelihood /= num_sweeps

        log_likelihood += plume_log_likelihood

        return log_likelihood
