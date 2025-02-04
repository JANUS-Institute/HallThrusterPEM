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

## Data conventions
The data used in the PEM is expected to be in a standard format.
This format may evolve over time to account for more data, but at present, when we read a CSV file, here is what we look for in the columns.
Note that we treat columns case-insensitvely, so `Anode current (A)` is treated the same as `anode current (a)`

### Operating conditions

Data is imported into a dictionary that maps operating conditions to data.
An Operating Condition (made concrete in the `OperatingCondition` class) consists of a unique set of an anode mass flow rate, a background pressure, and a discharge / anode voltage .
These quantities are **mandatory** for each data file, but can be provided in a few ways.
In cases where multiple options are allowed, the first matching column is chosen.

#### Background pressure
We expect a column named 'background pressure (torr)' (not case sensitive).
We assume the pressure is in units of Torr.

#### Anode mass flow rate
We look for the following column names in order:

1. A column named 'anode flow rate (mg/s)',
2. A column named 'total flow rate (mg/s)' and a column named 'anode-cathode flow ratio',
3. A column named 'total flow rate (mg/s)' and a column named 'cathode flow fraction'.

In all cases, the unit of the flow rate is expected to be mg/s.
For option 2, the cathode flow fraction is expected as a fraction between zero and one.
For option 3, the anode-cathode flow ratio is unitless and is expected to be greater than one.

#### Discharge voltage
We look for the following column names in order:

1. 'discharge voltage (v)',
2. 'anode voltage (v)'

In both cases, the unit of the voltage is expected to be Volts.

### Data

The following data-fields are all **optional**.
The `ThrusterData` struct will be populated only with what is provided.
For each of these quantities, an uncertainty can be provided, either relative or absolute.
The formats for uncertainties for a quantity of the form '{quantity} ({unit})' are
1. '{quantity} absolute uncertainty ({unit})'
2. '{quantity} relative uncertainty'

Relative uncertainties are fractions (so 0.2 == 20%) and absolute uncertainties are in the same units as the main quantity.

As an example, thrust of 100 mN and a relative uncertainty of 0.05 represents 100 +/- 5 mN.
We assume the errors are normally distributed about the nominal value with the uncertainty representing two standard deviations.
In this case, the distribution of the experimentally-measured thrust would be T ~ N(100, 2.5).
If both relative and absolute uncertainties are provided, we use the absolute uncertainty.
If an uncertainty is not provided, a relative uncertainty of 2% is assumed.

#### Thrust
We look for a column called 'thrust (mn)'.

#### Discharge current
We look for one of the following column names in order:

1. 'discharge current (a)'
2. 'anode current (a)'

#### Cathode coupling voltage
We look for a column called 'cathode coupling voltage (v)'

#### Ion current density
We look for three columns

1. The radial distance from the thruster exit plane.
Allowed keys: 'radial position from thruster exit (m)'

2. The angle relative to thruster centerline
Allowed keys: 'angular position from thruster centerline (deg)'

3. The current density.
Allowed keys: 'ion current density (ma/cm^2)'

We do not look for uncertainties for the radius and angle.
The current density is assumed to have units of mA / cm^2.
If one or two of these quantities is provided, we throw an error.

#### Ion velocity
We look for two columns

1. Axial position from anode
Allowed keys: 'axial position from anode (m)'

2. Ion velocity
Allowed keys: 'ion velocity (m/s)'

We do not look for uncertainties for the axial position.
The ion velocity is assumed to have units of m/s.
If only one of these quantities is provided, we throw an error.

"""  # noqa: E501

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Generic, Optional, Sequence, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.floating[Any]]
PathLike: TypeAlias = str | Path

opcond_keys_forward: dict[str, str] = {
    "p_b": "background_pressure_torr",
    "v_a": "discharge_voltage_v",
    "mdot_a": "anode_mass_flow_rate_kg_s",
}

opcond_keys_backward: dict[str, str] = {v: k for (k, v) in opcond_keys_forward.items()}


@dataclass(frozen=True)
class OperatingCondition:
    """Operating conditions for a Hall thruster. Currently includes background pressure (Torr),
    discharge voltage (V), and anode mass flow rate (kg/s).
    """

    background_pressure_torr: float
    discharge_voltage_v: float
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
    return -0.5 * np.sum(2 * np.log(std) + (mean - observation) ** 2 / (std**2))


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
            ion_velocity=IonVelocityData(
                axial_distance_m=outputs["u_ion_coords"][i],
                velocity_m_s=Measurement(outputs["u_ion"][i], np.full_like(outputs["u_ion"][i], NaN)),
            ),
            ion_current_sweeps=[
                CurrentDensitySweep(
                    radius_m=np.float64(1.0),
                    angles_rad=outputs["j_ion_coords"][i],
                    current_density_A_m2=Measurement(outputs["j_ion"][i], np.full_like(outputs["j_ion"][i], NaN)),
                )
            ],
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
    )

    # Contribution to likelihood from ion velocity
    if data.ion_velocity is not None and observation.ion_velocity is not None:
        log_likelihood += _interp_gauss_logpdf(
            data.ion_velocity.axial_distance_m,
            data.ion_velocity.velocity_m_s,
            observation.ion_velocity.axial_distance_m,
            observation.ion_velocity.velocity_m_s,
        )

    # Contribution to likelihood from ion current density
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

            log_likelihood += _interp_gauss_logpdf(
                data_sweep.angles_rad,
                data_sweep.current_density_A_m2,
                observation_sweep.angles_rad,
                observation_sweep.current_density_A_m2,
            )

    return log_likelihood


def rel_uncertainty_key(qty: str) -> str:
    body, _ = qty.split('(', 1) if '(' in qty else (qty, '')
    return body.rstrip().casefold() + ' relative uncertainty'


def abs_uncertainty_key(qty: str) -> str:
    body, unit = qty.split('(', 1) if '(' in qty else (qty, '')
    return body.rstrip().casefold() + ' absolute uncertainty (' + unit


def read_measurement(
    table: dict[str, Array],
    key: str,
    val: Array,
    start: int = 0,
    end: int | None = None,
    scale: float = 1.0,
    scalar: bool = False,
) -> Measurement:
    default_rel_err = 0.02
    mean = val * scale
    std = default_rel_err * mean / 2

    if (abs_err := table.get(abs_uncertainty_key(key))) is not None:
        std = abs_err * scale / 2
    elif (rel_err := table.get(rel_uncertainty_key(key))) is not None:
        std = rel_err * mean / 2

    if end is None:
        end = val.size

    if scalar:
        return Measurement(mean[start], std[start])
    else:
        return Measurement(mean[start:end], std[start:end])


def load_single_file(file: PathLike) -> dict[OperatingCondition, ThrusterData]:
    """Load data from a single file into a dict map of `OperatingCondition` -> `ThrusterData`."""

    # Read table from file
    table = _table_from_file(file, delimiter=",", comments="#")
    keys = list(table.keys())
    data: dict[OperatingCondition, ThrusterData] = {}

    # Get anode mass flow rate (or compute it from total flow rate and cathode flow fraction or anode flow ratio)
    mdot_a = table.get("anode flow rate (mg/s)")
    if mdot_a is None:
        mdot_t = table.get("total flow rate (mg/s)")
        if mdot_t is not None and (cathode_flow_fraction := table.get("cathode flow fraction")) is not None:
            anode_flow_fraction = 1 - cathode_flow_fraction
            mdot_a = mdot_t * anode_flow_fraction
        elif mdot_t is not None and (anode_flow_ratio := table.get("anode-cathode flow ratio")) is not None:
            anode_flow_fraction = anode_flow_ratio / (anode_flow_ratio + 1)
            mdot_a = mdot_t * anode_flow_fraction
        else:
            raise KeyError(
                f"{file}: No mass flow rate provided."
                + " Expected a key called `anode flow rate (mg/s)` or a key called [`total flow rate (mg/s)`"
                + "and one of (`anode-cathode flow ratio`, `cathode flow fraction`)]"
            )
    mdot_a *= 1e-6  # convert flow rate from mg/s to kg/s

    # Get background pressure
    P_B = table.get("background pressure (torr)")
    if P_B is None:
        raise KeyError(f"{file}: No background pressure provided. Expected a key called `background pressure (torr)`.")

    # Get discharge voltage
    V_a = table.get("discharge voltage (v)", table.get("anode voltage (v)", None))
    if V_a is None:
        raise KeyError(
            f"{file}: No discharge voltage provided."
            + " Expected a key called `discharge voltage (v)` or `anode voltage (v)`."
        )

    num_rows = len(table[keys[0]])
    row = 0
    opcond_start = 0
    opcond = OperatingCondition(P_B[0], V_a[0], mdot_a[0])

    while True:
        next_opcond = opcond
        if row < num_rows:
            next_opcond = OperatingCondition(P_B[row], V_a[row], mdot_a[row])
            if next_opcond == opcond:
                row += 1
                continue

        # either at end of operating condition or end of table
        # fill up thruster data object for this row
        data[opcond] = ThrusterData()

        # Fill up data
        # We assume all errors (expressed as value +/- error) correspond to two standard deviations
        # We also assume a default relative error of 2%
        for key, val in table.items():
            if key == "thrust (mn)":
                data[opcond].thrust_N = read_measurement(table, key, val, opcond_start, scale=1e-3, scalar=True)

            elif key in {"anode current (a)", "discharge current (a)"}:
                data[opcond].discharge_current_A = read_measurement(table, key, val, opcond_start, scalar=True)

            elif key == "cathode coupling voltage (v)":
                data[opcond].cathode_coupling_voltage_V = read_measurement(table, key, val, opcond_start, scalar=True)

            elif key == "ion velocity (m/s)":
                data[opcond].ion_velocity = IonVelocityData(
                    axial_distance_m=table["axial position from anode (m)"][opcond_start:row],
                    velocity_m_s=read_measurement(table, key, val, start=opcond_start, end=row, scalar=False),
                )

            elif key == "ion current density (ma/cm^2)":
                # Load ion current density data
                jion = read_measurement(table, key, val, start=opcond_start, end=row, scale=10, scalar=False)
                radii_m = table["radial position from thruster exit (m)"][opcond_start:row]
                jion_coords: Array = table["angular position from thruster centerline (deg)"][opcond_start:row]
                unique_radii = np.unique(radii_m)
                sweeps: list[CurrentDensitySweep] = []
                # Keep only angles between 0 and 90 degrees
                keep_inds = np.logical_and(jion_coords < 90, jion_coords >= 0)

                for r in unique_radii:
                    inds = np.logical_and(radii_m == r, keep_inds)
                    sweeps.append(
                        CurrentDensitySweep(
                            radius_m=r,
                            angles_rad=jion_coords[inds] * np.pi / 180,
                            current_density_A_m2=Measurement(mean=jion.mean[inds], std=jion.std[inds]),
                        )
                    )

                current_sweeps = data[opcond].ion_current_sweeps
                if current_sweeps is None:
                    data[opcond].ion_current_sweeps = sweeps
                else:
                    data[opcond].ion_current_sweeps = current_sweeps + sweeps

        # Advance to next operating condition or break out of loop if we're at the end of the table
        if row == num_rows:
            break

        opcond = next_opcond
        opcond_start = row

    return data


def update_data(
    old_data: dict[OperatingCondition, ThrusterData], new_data: dict[OperatingCondition, ThrusterData]
) -> dict[OperatingCondition, ThrusterData]:
    data = old_data.copy()
    for opcond in new_data.keys():
        if opcond in old_data.keys():
            data[opcond] = ThrusterData.update(old_data[opcond], new_data[opcond])
        else:
            data[opcond] = new_data[opcond]

    return data


def load(files: Sequence[PathLike] | PathLike) -> dict[OperatingCondition, ThrusterData]:
    """Load all data from the given files into a dict map of `OperatingCondition` -> `ThrusterData`.
    Each thruster operating condition corresponds to one set of thruster measurements or quantities of interest (QoIs).

    :param files: A list of file paths or a single file path to load data from (only .csv supported).
    :return: A dict map of `OperatingCondition` -> `ThrusterData` objects.
    """
    data: dict[OperatingCondition, ThrusterData] = {}
    if isinstance(files, list):
        # Recursively load resources in this list (possibly list of lists)
        for file in files:
            new_data = load(file)
            data = update_data(data, new_data)
    else:
        new_data = load_single_file(files)
        data = update_data(data, new_data)

    return data


def _table_from_file(file: PathLike, delimiter=",", comments="#") -> dict[str, Array]:
    """Return a `dict` of `numpy` arrays from a CSV file. The keys of the dict are the column names in the CSV."""
    # Read header of file to get column names
    # We skip comments (lines starting with the string in the `comments` arg)
    header_start = 0
    header = ""
    with open(file, "r", encoding="utf-8-sig") as f:
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
