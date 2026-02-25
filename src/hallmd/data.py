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

from pem_core.data import UNITS

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
