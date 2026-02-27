"""The `hallmd.data` package contains utilities for loading and processing Hall thruster experimental data.
Example data for the SPT-100 thruster lives in the [pem_data repo](https://github.com/JANUS-Institute/pem_dat).

## Data conventions
The data used in the Hall Thruster PEM is expected to be in somewhat standard format.
This format may evolve over time to account for more data, but at present, when we read a CSV file, here is what we look for in the columns.
Note that we treat columns case-insensitively (except the units), so `Anode current (A)` is treated the same as `anode current (A)`.
Additionally, we automatically convert units to SI as best we can.

### Operating conditions

Data is imported into a dictionary that maps operating conditions to data.
An operating condition consists of a unique set of an anode mass flow rate, a background pressure, a discharge / anode voltage, and a magnetic field scale.
The flow rate and voltage are mandatory, while the pressure and field scale are optional (and assumed to be 0 and 1, respectively).
These can be provided in a few ways.
In cases where multiple options are allowed, the first matching column is chosen.

#### Background pressure
We expect a column named 'background pressure (torr)' (not case sensitive).
We assume the pressure is in units of Torr.

#### Anode mass flow rate
We look for the following column names in order:

1. A column named 'anode flow rate (mg/s)',
2. A column named 'total flow rate (mg/s)' and a column named 'anode-cathode flow ratio',
3. A column named 'total flow rate (mg/s)' and a column named 'cathode flow fraction'.

In all cases, the unit of the flow rate is expected to be mass / time.
For option 2, the cathode flow fraction is expected as a fraction between zero and one.
For option 3, the anode-cathode flow ratio is unitless and is expected to be greater than one.

#### Discharge voltage
We look for the following column names in order:

1. 'discharge voltage (v)',
2. 'anode voltage (v)'

In both cases, the unit of the voltage is expected to be Volts.

### Data

The following data-fields are all **optional**.
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
We look for a column called 'thrust (mn)' or 'thrust (n)'.
We then convert the thrust to Newtons internally.

#### Discharge current
We look for one of the following column names in order:

1. 'discharge current (a)'
2. 'anode current (a)'

#### Cathode coupling voltage
We look for a column called 'cathode coupling voltage (v)'.

#### Ion current density
We look for three columns

1. The radial distance from the thruster exit plane.
Allowed keys: 'radial position from thruster exit (m)'

2. The angle relative to thruster centerline
Allowed keys: 'angular position from thruster centerline (deg)'

3. The current density.
Allowed keys: 'ion current density (ma/cm^2)' or 'ion current density (a/m^2)'

We do not look for uncertainties for the radius and angle.
The current density is assumed to have units of mA / cm^2 or A / m^2, depending on the key.
If one or two of these quantities is provided, we throw an error.

#### Ion velocity
We look for two columns:

1. Axial position from anode
Allowed keys: 'axial position from anode (m)', 'axial distance from anode (m)'

2. Ion velocity
Allowed keys: 'ion velocity (m/s)'

We do not look for uncertainties for the axial position.
The ion velocity is assumed to have units of m/s.
If only one of these quantities is provided, we throw an error.

"""  # noqa: E501

from pem_core.data import UNITS, DataEntry, DataInstance, DataField
import numpy as np
import xarray as xr

#==================================================================================================
# This section defines the operating conditions and QoIs of the Hall thruster PEM
#==================================================================================================

UNITS.define("Torr = 133.322368 pascal = Torr")

HT_OP_VARS = {
    "discharge voltage": {
        "unit": UNITS.volts,
    },
    "anode mass flow rate": {
        "unit": UNITS.kg / UNITS.second,
    },
    "background pressure": {
        "unit": UNITS.torr,
        "default": 0.0,
    },
    "magnetic field scale": {
        "unit": UNITS.dimensionless,
        "default": 1.0,
    },
}

HT_COORDS = {
    "z": UNITS.meter,
    "r": UNITS.meter,
    "theta": UNITS.rad,
}

HT_QOIS = {
    "cathode coupling voltage": {
        "unit": UNITS.volts,
    },
    "discharge current": {
        "unit": UNITS.ampere,
    },
    "thrust": {
        "unit": UNITS.newton,
    },
    "ion velocity": {
        "unit": UNITS.meter / UNITS.second,
        "coords": ("z",),
    },
    "ion current density": {
        "unit": UNITS.ampere / UNITS.meter**2,
        "coords": ("r", "theta"),
    },
}

FLOW_RATE_KEY = "anode mass flow rate"

HT_RENAME_MAP = {
    "anode voltage" : "discharge voltage",
    "anode current" : "discharge current",
    "anode flow rate" : FLOW_RATE_KEY,
    "axial distance from anode": "z",
    "axial position from anode": "z",
    "axial ion velocity": "ion velocity",
    "angular position from thruster centerline": "theta",
    "radial position from thruster exit": "r",
}

OPCOND_SHORT_NAMES = {
    "discharge voltage": "V_a",
    "anode mass flow rate": "mdot_a",
    "background pressure": "P_b",
    "magnetic field scale": "B_hat",
}

#==================================================================================================
# This section contains further utilities for working with the Hall thruster PEM and data
#==================================================================================================

def pem_to_xarray(
        operating_conditions: list[dict[str, float]],
        outputs: dict, sweep_radii: np.ndarray,
        use_corrected_thrust: bool = True
    ) -> list[DataEntry]:
    """Convert the outputs of the Hall thruster PEM to xarrays so that we can compare them to data"""

    data_entries: list[DataEntry] = []

    for (i, opcond) in enumerate(operating_conditions):

        if use_corrected_thrust:
            # With multiple radii, we have multiple thrusts. Pick the last one as sweep_radii are sorted.
            thrust = xr.DataArray(np.atleast_1d(outputs['T_c'][i])[-1])
        else:
            thrust = xr.DataArray(outputs['T'][i])

        Id = xr.DataArray(outputs['I_d'][i])
        Vcc = xr.DataArray(outputs['V_cc'][i])

        z = outputs['u_ion_coords'][i]
        uion = outputs['u_ion'][i]
        uion_arr = xr.DataArray(uion, coords=[z], dims=["z"])

        theta = outputs['j_ion_coords'][i]
        r = sweep_radii
        jion = np.atleast_3d(outputs['j_ion'])[i, :, :].T
        jion_arr = xr.DataArray(jion, coords=[r, theta], dims=["r", "theta"])

        instance: DataInstance = {
            "discharge current": DataField(val=Id, unit="A"),
            "cathode coupling voltage": DataField(val=Vcc, unit="V"),
            "thrust": DataField(val=thrust, unit="N"),
            "ion velocity": DataField(val=uion_arr, unit="m/s"),
            "ion current density": DataField(val=jion_arr, unit="A/m^2"),
        }

        entry = DataEntry(operating_condition=opcond, data=instance)
        data_entries.append(entry)

    return data_entries