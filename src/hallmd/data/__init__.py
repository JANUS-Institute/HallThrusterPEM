"""The `hallmd.data` package contains a folder for each unique thruster. The experimental data for each thruster
is further divided by folders for each individual paper or reference. The raw experimental data is contained within
these folders in any supported format (currently only .csv). Any additional documentation
for the datasets is encouraged (e.g. citations, descriptions, summaries, etc.) and can be included in the data folders.

## Thrusters

### SPT-100
Currently the only thruster with available data. Data for the SPT-100 comes from four sources:

1. [Diamant et al. 2014](https://arc.aiaa.org/doi/10.2514/6.2014-3710) - provides thrust and ion current density data as a function of chamber background pressure.
2. [Macdonald et al. 2019](https://arc.aiaa.org/doi/10.2514/1.B37133) - provides ion velocity profiles for varying chamber pressures.
3. [Sankovic et al. 1993](https://www.semanticscholar.org/paper/Performance-evaluation-of-the-Russian-SPT-100-at-Sankovic-Hamley/81b7d985669b21aa1a8419277c52e7a879bf3b46) - provides thrust at varying operating conditions.
4. [Jorns and Byrne. 2021](https://pepl.engin.umich.edu/pdf/2021_PSST_Jorns.pdf) - provides cathode coupling voltages at same conditions as Diamant et al. 2014.

Citations:
``` title="SPT-100.bib"
--8<-- "hallmd/data/spt100/spt100.bib:citations"
```
"""  # noqa: E501

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Generic, Optional, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.floating[Any]]
PathLike: TypeAlias = str | Path


@dataclass(frozen=True)
class OperatingCondition:
    """Operating conditions for a Hall thruster. Currently includes background pressure (Torr),
    discharge voltage (V), and anode mass flow rate (kg/s).
    """
    background_pressure_Torr: float
    discharge_voltage_V: float
    anode_mass_flow_rate_kg_s: float


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
    ion_velocity_coords_m: Optional[Array] = None
    ion_velocity_m_s: Optional[Measurement[Array]] = None
    # Plume
    divergence_angle_rad: Optional[Measurement[np.float64]] = None
    ion_current_density_radius_m: Optional[float] = None
    ion_current_density_coords_m: Optional[Array] = None
    ion_current_density_A_m2: Optional[Measurement[Array]] = None

    def __str__(self) -> str:
        fields_str = ",\n".join([f"\t{field.name} = {val}" for field in fields(ThrusterData)
                                 if (val := getattr(self, field.name)) is not None])
        return f"ThrusterData(\n{fields_str}\n)\n"


def pem_to_thrusterdata(
    operating_conditions: list[OperatingCondition], outputs
) -> dict[OperatingCondition, ThrusterData]:
    # Assemble output dict from operating conditions -> results
    # Note, we assume that the pem outputs are ordered based on the input operating conditions
    NaN = np.float64(np.nan)
    output_dict = {
        opcond: ThrusterData(
            cathode_coupling_voltage_V=Measurement(outputs["V_cc"][i], NaN),
            thrust_N=Measurement(outputs["T"][i], NaN),
            discharge_current_A=Measurement(outputs["I_d"][i], NaN),
            ion_current_A=Measurement(outputs["I_B0"][i], NaN),
            ion_velocity_coords_m=outputs["u_ion_coords"][i],
            ion_velocity_m_s=Measurement(outputs["u_ion"][i], np.full_like(outputs["u_ion"][i], NaN)),
            ion_current_density_coords_m=outputs["j_ion_coords"][i],
            ion_current_density_A_m2=Measurement(outputs["j_ion"][i], np.full_like(outputs["j_ion"][i], NaN)),
            ion_current_density_radius_m=1.0,
            efficiency_mass=Measurement(outputs["eta_m"][i], NaN),
            efficiency_current=Measurement(outputs["eta_c"][i], NaN),
            efficiency_voltage=Measurement(outputs["eta_v"][i], NaN),
            efficiency_anode=Measurement(outputs["eta_a"][i], NaN),
        )
        for (i, opcond) in enumerate(operating_conditions)
    }

    return output_dict


def log_likelihood(data: ThrusterData, observation: ThrusterData) -> np.float64:
    log_likelihood = (
        # Add contributions from global performance metrics
        _measurement_gauss_logpdf(data.cathode_coupling_voltage_V, observation.cathode_coupling_voltage_V)
        + _measurement_gauss_logpdf(data.thrust_N, observation.thrust_N)
        + _measurement_gauss_logpdf(data.discharge_current_A, observation.discharge_current_A)
        + _measurement_gauss_logpdf(data.ion_current_A, observation.ion_current_A)
        + _measurement_gauss_logpdf(data.efficiency_current, observation.efficiency_current)
        + _measurement_gauss_logpdf(data.efficiency_mass, observation.efficiency_mass)
        + _measurement_gauss_logpdf(data.efficiency_voltage, observation.efficiency_voltage)
        + _measurement_gauss_logpdf(data.efficiency_anode, observation.efficiency_anode)
        + _measurement_gauss_logpdf(data.divergence_angle_rad, observation.divergence_angle_rad)
        # interpolated average pointwise error from ion velocity and ion current density
        + _interp_gauss_logpdf(
            data.ion_velocity_coords_m,
            data.ion_velocity_m_s,
            observation.ion_velocity_coords_m,
            observation.ion_velocity_m_s,
        )
        + _interp_gauss_logpdf(
            data.ion_current_density_coords_m,
            data.ion_current_density_A_m2,
            observation.ion_current_density_coords_m,
            observation.ion_current_density_A_m2,
        )
    )
    return log_likelihood


def load(files: list[PathLike] | PathLike) -> dict[OperatingCondition, ThrusterData]:
    """Load all data from the given files into a dict map of `OperatingCondition` -> `ThrusterData`.
    Each thruster operating condition corresponds to one set of thruster measurements or quantities of interest (QoIs).

    :param files: A list of file paths or a single file path to load data from (only .csv supported).
    :return: A dict map of `OperatingCondition` -> `ThrusterData` objects.
    """
    data: dict[OperatingCondition, ThrusterData] = {}
    if isinstance(files, list):
        # Recursively load resources in this list (possibly list of lists)
        for file in files:
            data.update(load(file))
    else:
        data.update(_load_single(files))

    return data


def _load_single(file: PathLike) -> dict[OperatingCondition, ThrusterData]:
    """Load data from a single file into a dict map of `OperatingCondition` -> `ThrusterData`."""
    if not Path(file).suffix == '.csv':
        raise ValueError(f"Unsupported file format: {Path(file).suffix}. Only .csv files are supported.")

    table = _table_from_file(file, delimiter=",", comments="#")
    data: dict[OperatingCondition, ThrusterData] = {}

    # Compute anode mass flow rate, if not present
    mdot_a_key = "anode flow rate (mg/s)"
    mdot_t_key = "total flow rate (mg/s)"
    flow_ratio_key = "anode-cathode flow ratio"
    keys = list(table.keys())

    if mdot_a_key in keys:
        mdot_a = table[mdot_a_key] * 1e-6  # convert to kg/s
    else:
        if mdot_t_key in keys and flow_ratio_key in keys:
            flow_ratio = table[flow_ratio_key]
            anode_flow_fraction = flow_ratio / (flow_ratio + 1)
            mdot_a = table[mdot_t_key] * anode_flow_fraction * 1e-6  # convert to kg/s
        else:
            raise KeyError(
                f"{file}: No mass flow rate provided."
                + " Need either `anode flow rate (mg/s)` or [`total flow rate (mg/s)` and `anode-cathode flow ratio`]"
            )

    # Get background pressure and discharge voltage
    P_B = table["background pressure (torr)"]
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
    """Return a `dict` of `numpy` arrays from a CSV file. The keys of the dict are the column names in the CSV."""
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
