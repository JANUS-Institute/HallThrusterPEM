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
    P_B: float
    V_a: float
    mdot_a: float


T = TypeVar("T")
T2 = TypeVar("T2")


@dataclass(frozen=True)
class Measurement(Generic[T]):
    mean: T
    std: T

    def __str__(self):
        return f"(μ = {self.mean}, σ = {self.std})"


def _gauss_logpdf[T: np.float64 | Array](mean: T, std: T, observation: T) -> np.float64:
    var = std**2
    term1 = np.mean(np.log(2 * np.pi * var)) / 2
    term2 = np.mean((mean - observation) ** 2 / (2 * var))
    return -term1 - term2


def _measurement_gauss_logpdf[T: np.float64 | Array](
    data: Measurement[T] | None, observation: Measurement[T] | None
) -> np.float64:
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
    V_cc: Measurement[np.float64] | None = None
    # Thruster
    T: Measurement[np.float64] | None = None
    I_D: Measurement[np.float64] | None = None
    I_B0: Measurement[np.float64] | None = None
    eta_c: Measurement[np.float64] | None = None
    eta_m: Measurement[np.float64] | None = None
    eta_v: Measurement[np.float64] | None = None
    eta_a: Measurement[np.float64] | None = None
    uion_coords: Array | None = None
    uion: Measurement[Array] | None = None
    # Plume
    div_angle: Measurement[np.float64] | None = None
    jion_radius: float | None = None
    jion_coords: Array | None = None
    jion: Measurement[Array] | None = None

    def log_likelihood_of(self, observation: "ThrusterData") -> np.float64:
        # Add contributions from global performance metrics
        log_likelihood = (
            _measurement_gauss_logpdf(self.V_cc, observation.V_cc)
            + _measurement_gauss_logpdf(self.T, observation.T)
            + _measurement_gauss_logpdf(self.I_D, observation.I_D)
            + _measurement_gauss_logpdf(self.I_B0, observation.I_B0)
            + _measurement_gauss_logpdf(self.eta_c, observation.eta_c)
            + _measurement_gauss_logpdf(self.eta_m, observation.eta_m)
            + _measurement_gauss_logpdf(self.eta_v, observation.eta_v)
            + _measurement_gauss_logpdf(self.eta_a, observation.eta_a)
            + _measurement_gauss_logpdf(self.div_angle, observation.div_angle)
            + _interp_gauss_logpdf(self.uion_coords, self.uion, observation.uion_coords, observation.uion)
        )

        if self.jion_radius is not None and self.jion_radius == observation.jion_radius:
            log_likelihood += _interp_gauss_logpdf(
                self.jion_coords, self.jion, observation.jion_coords, observation.jion
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
                data[opcond].T = Measurement(mean=T, std=T_std)

            elif key == "anode current (a)":
                # Load discharge current data
                # assume a 0.1-A std deviation for discharge current
                data[opcond].I_D = Measurement(mean=val[opcond_start_row], std=0.1)

            elif key == "cathode coupling voltage (v)":
                # Load cathode coupling data
                V_cc = val[opcond_start_row]
                V_cc_std = table["cathode coupling voltage absolute uncertainty (v)"][opcond_start_row] / 2
                data[opcond].V_cc = Measurement(mean=V_cc, std=V_cc_std)

            elif key == "ion velocity (m/s)":
                # Load ion velocity data
                uion: Array = val[opcond_start_row:row_num]
                uion_std: Array = table["ion velocity absolute uncertainty (m/s)"][opcond_start_row:row_num] / 2
                data[opcond].uion = Measurement(mean=uion, std=uion_std)
                data[opcond].uion_coords = table["axial position from anode (m)"][opcond_start_row:row_num]

            elif key == "ion current density (ma/cm^2)":
                # Load ion current density data
                jion: Array = val[opcond_start_row:row_num] * 10  # Convert to A / m^2
                jion_std: Array = table["ion current density relative uncertainty"][opcond_start_row:row_num] * jion / 2
                r = table["radial position from thruster exit (m)"][0]
                jion_coords: Array = table["angular position from thruster centerline (deg)"][opcond_start_row:row_num]

                # Keep only measurements at angles less than 90 degrees
                keep_inds = jion_coords < 90
                data[opcond].jion_coords = jion_coords[keep_inds] * np.pi / 180
                data[opcond].jion_radius = r
                data[opcond].jion = Measurement(mean=jion[keep_inds], std=jion_std[keep_inds])

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
