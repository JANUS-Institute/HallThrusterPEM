"""The `hallmd.data` package contains a folder for each unique thruster. The experimental data for each thruster
is further divided by folders for each individual paper or reference. The raw experimental data is contained within
these folders in any arbitrary format (h5, json, csv, etc.). Each set of raw experimental data should come with a
`dataloader.py` file that reads from the raw data into standardized Python objects. Any additional documentation
for the datasets is encouraged (e.g. citations, descriptions, summaries, etc.) and can be included in the data folders.
## Thrusters

### SPT-100
Currently the only thruster with available data. Data for the SPT-100 comes from four sources:

1. [Diamant et al. 2014](https://arc.aiaa.org/doi/10.2514/6.2014-3710) - provides thrust, cathode coupling voltage, and ion current density data as a function of chamber background pressure.
2. [Macdonald et al. 2019](https://arc.aiaa.org/doi/10.2514/1.B37133) - provides ion velocity profiles for varying chamber pressures.
3. [Sankovic et al. 1993](https://www.semanticscholar.org/paper/Performance-evaluation-of-the-Russian-SPT-100-at-Sankovic-Hamley/81b7d985669b21aa1a8419277c52e7a879bf3b46) - provides thrust at varying operating conditions.
4. [Jorns and Byrne. 2021](https://pepl.engin.umich.edu/pdf/2021_PSST_Jorns.pdf) - provides cathode coupling voltages at same conditions as Diamant et al. 2024.

Citations:
``` title="SPT-100.bib"
--8<-- "hallmd/data/spt100/spt100.bib:citations"
```
"""  # noqa: E501

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Generic, Sequence, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.floating[Any]]
PathLike: TypeAlias = str | Path


@dataclass(frozen=True)
class OperatingCondition:
    background_pressure_Torr: float
    discharge_voltage_V: float
    anode_mass_flow_rate_kg_s: float


T = TypeVar("T", np.float64, Array)


@dataclass(frozen=True)
class Measurement(Generic[T]):
    mean: T
    std: T

    def __str__(self):
        return f"(μ = {self.mean}, σ = {self.std})"


def _gauss_logpdf(mean: T, std: T, observation: T) -> np.float64:
    var = std**2
    term1 = np.mean(np.log(2 * np.pi * var)) / 2
    term2 = np.mean((mean - observation) ** 2 / (2 * var))
    return -term1 - term2


def _measurement_gauss_logpdf(data: Measurement[T] | None, observation: Measurement[T] | None) -> np.float64:
    if data is None or observation is None:
        return np.float64(0.0)

    return _gauss_logpdf(data.mean, data.std, observation.mean)


def _interp_gauss_logpdf(
    coords: Array | None,
    data: Measurement[Array] | None,
    obs_coords: Array | None,
    observation: Measurement[Array] | None,
) -> np.float64:
    if coords is None or data is None or obs_coords is None or observation is None:
        return np.float64(0.0)

    obs_interp_mean = np.interp(coords, obs_coords, observation.mean)
    obs_interp = Measurement(obs_interp_mean, np.zeros(1))
    return _measurement_gauss_logpdf(data, obs_interp)


@dataclass
class ThrusterData:
    # Cathode
    cathode_coupling_voltage_V: Measurement[np.float64] | None = None
    # Thruster
    thrust_N: Measurement[np.float64] | None = None
    discharge_current_A: Measurement[np.float64] | None = None
    ion_current_A: Measurement[np.float64] | None = None
    efficiency_current: Measurement[np.float64] | None = None
    efficiency_mass: Measurement[np.float64] | None = None
    efficiency_voltage: Measurement[np.float64] | None = None
    efficiency_anode: Measurement[np.float64] | None = None
    ion_velocity_coords_m: Array | None = None
    ion_velocity_m_s: Measurement[Array] | None = None
    # Plume
    divergence_angle_rad: Measurement[np.float64] | None = None
    ion_current_density_radius_m: float | None = None
    ion_current_density_coords_m: Array | None = None
    ion_current_density_A_m2: Measurement[Array] | None = None

    def log_likelihood_of(self, observation: "ThrusterData") -> np.float64:
        # Add contributions from global performance metrics
        log_likelihood = (
            _measurement_gauss_logpdf(self.cathode_coupling_voltage_V, observation.cathode_coupling_voltage_V)
            + _measurement_gauss_logpdf(self.thrust_N, observation.thrust_N)
            + _measurement_gauss_logpdf(self.discharge_current_A, observation.discharge_current_A)
            + _measurement_gauss_logpdf(self.ion_current_A, observation.ion_current_A)
            + _measurement_gauss_logpdf(self.efficiency_current, observation.efficiency_current)
            + _measurement_gauss_logpdf(self.efficiency_mass, observation.efficiency_mass)
            + _measurement_gauss_logpdf(self.efficiency_voltage, observation.efficiency_voltage)
            + _measurement_gauss_logpdf(self.efficiency_anode, observation.efficiency_anode)
            + _measurement_gauss_logpdf(self.divergence_angle_rad, observation.divergence_angle_rad)
            + _interp_gauss_logpdf(
                self.ion_velocity_coords_m,
                self.ion_velocity_m_s,
                observation.ion_velocity_coords_m,
                observation.ion_velocity_m_s,
            )
        )

        if (
            self.ion_current_density_radius_m is not None
            and self.ion_current_density_radius_m == observation.ion_current_density_radius_m
        ):
            log_likelihood += _interp_gauss_logpdf(
                self.ion_current_density_coords_m,
                self.ion_current_density_A_m2,
                observation.ion_current_density_coords_m,
                observation.ion_current_density_A_m2,
            )

        return log_likelihood

    def __str__(self) -> str:
        indent = "\t"
        out: str = "ThrusterData(\n"
        for field in fields(ThrusterData):
            val = getattr(self, field.name)
            if val is not None:
                out += f"{indent}{field.name} = {val},\n"

        out += ")\n"
        return out


def load(files: Sequence[PathLike] | PathLike) -> dict[OperatingCondition, ThrusterData]:
    data: dict[OperatingCondition, ThrusterData] = {}
    if isinstance(files, Sequence):
        # Recursively load resources in this list (possibly list of lists)
        for file in files:
            data.update(load(file))
    else:
        data.update(_load_dataset(files))

    return data


def _load_dataset(file: PathLike) -> dict[OperatingCondition, ThrusterData]:
    table = _table_from_file(file, delimiter=",", comments="#")
    data: dict[OperatingCondition, ThrusterData] = {}

    # Compute anode mass flow rate, if not present
    mdot_a_key = "anode flow rate (mg/s)"
    mdot_t_key = "total flow rate (mg/s)"
    flow_ratio_key = "anode-cathode flow ratio"
    keys = list(table.keys())

    if mdot_a_key in keys:
        mdot_a = table[mdot_a_key]
    else:
        if mdot_t_key in keys and flow_ratio_key in keys:
            flow_ratio = table[flow_ratio_key]
            anode_flow_fraction = flow_ratio / (flow_ratio + 1)
            mdot_a = table[mdot_t_key] * anode_flow_fraction
        else:
            raise KeyError(
                f"{file}: No mass flow rate provided."
                + " Need either `anode flow rate (mg/s)` or [`total flow rate (mg/s)` and `anode-cathode flow ratio`]"
            )

    # Get background pressure and discharge voltage
    P_B = np.log10(table["background pressure (torr)"])
    V_a = table["anode voltage (v)"]

    num_rows = len(table[keys[0]])
    row_num = 0
    opcond_start_row = 0
    opcond = OperatingCondition(P_B[0], V_a[0], mdot_a[0])

    while True:
        next_opcond = opcond
        if row_num < num_rows:
            next_opcond = OperatingCondition(P_B[row_num], V_a[row_num], mdot_a[row_num])
            if next_opcond == opcond:
                row_num += 1
                continue

        # either at end of operating condition or end of table
        # fill up thruster data object for this row
        data[opcond] = ThrusterData()

        # Fill up data
        # We assume all errors (expressed as value +/- error) correspond to two standard deviations
        for key, val in table.items():
            if key == "thrust (mn)":
                # Load thrust data
                T = val[opcond_start_row] * 1e-3  # convert to Newtons
                T_std = table["thrust relative uncertainty"][opcond_start_row] * T / 2
                data[opcond].thrust_N = Measurement(mean=T, std=T_std)

            elif key == "anode current (a)":
                # Load discharge current data
                # assume a 0.1-A std deviation for discharge current
                data[opcond].discharge_current_A = Measurement(mean=val[opcond_start_row], std=np.float64(0.1))

            elif key == "cathode coupling voltage (v)":
                # Load cathode coupling data
                V_cc = val[opcond_start_row]
                V_cc_std = table["cathode coupling voltage absolute uncertainty (v)"][opcond_start_row] / 2
                data[opcond].cathode_coupling_voltage_V = Measurement(mean=V_cc, std=V_cc_std)

            elif key == "ion velocity (m/s)":
                # Load ion velocity data
                uion: Array = val[opcond_start_row:row_num]
                uion_std: Array = table["ion velocity absolute uncertainty (m/s)"][opcond_start_row:row_num] / 2
                data[opcond].ion_velocity_m_s = Measurement(mean=uion, std=uion_std)
                data[opcond].ion_velocity_coords_m = table["axial position from anode (m)"][opcond_start_row:row_num]

            elif key == "ion current density (ma/cm^2)":
                # Load ion current density data
                jion: Array = val[opcond_start_row:row_num] * 10  # Convert to A / m^2
                jion_std: Array = table["ion current density relative uncertainty"][opcond_start_row:row_num] * jion / 2
                r = table["radial position from thruster exit (m)"][0]
                jion_coords: Array = table["angular position from thruster centerline (deg)"][opcond_start_row:row_num]

                # Keep only measurements at angles less than 90 degrees
                keep_inds = jion_coords < 90
                data[opcond].ion_current_density_coords_m = jion_coords[keep_inds] * np.pi / 180
                data[opcond].ion_current_density_radius_m = r
                data[opcond].ion_current_density_A_m2 = Measurement(mean=jion[keep_inds], std=jion_std[keep_inds])

        # Advance to next operating condition or break out of loop if we're at the end of the table
        if row_num == num_rows:
            break

        opcond = next_opcond
        opcond_start_row = row_num

    return data


def _table_from_file(file: PathLike, delimiter=",", comments="#") -> dict[str, Array]:
    # Read header of file to get column names
    # We skip comments (lines starting with the string in the `comments` arg)
    header_start = 0
    header = ""
    with open(file, "r") as f:
        for i, line in enumerate(f):
            if not line.startswith(comments):
                header = line.rstrip()
                header_start = i
                break

    if header == "":
        return {}

    column_names = header.split(delimiter)
    data = np.genfromtxt(file, delimiter=delimiter, comments=comments, skip_header=header_start + 1)

    table: dict[str, Array] = {column.casefold(): data[:, i] for (i, column) in enumerate(column_names)}
    return table
