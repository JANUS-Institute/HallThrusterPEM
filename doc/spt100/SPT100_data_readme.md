# SPT-100 Hall thruster data for use in prototype predictive engineering model (PEM)

## Data sources

The data for the SPT-100 in this directory comes from three papers, listed below.
With one exception (described below), the data from each paper is contained within a directory whose name is given in brackets after each paper's title.

1. J. Sankovic, J. Hamley, and T. Haag. Performance Evaluation of the Russian SPT-100 Thruster at NASA LeRC (1993) [SHH-1993]
2. S. Bechu et al. Comparison between two kinds of Hall Thrusters: SPT100 and ATON (2000). [SB-2000]
3. N. Dorval et al. Determination of the ionization and acceleration zones in a stationary plasma thruster by optical spectroscopy study: Experiments and model (2002) [ND-2002] 
4. N. MacDonald-Tenenbaum, M. Holmes, and W. Hargus, Jr. Background Pressure Effects on Ion Velocity Distributions in an SPT-100 Hall Thruster (2019) [MHH-2019]
5. K. Diamant, R. Liang, and R. Corey The Effect of Background Pressure on SPT-100 Hall Thruster Performance. [DLC-2014]

Paper 3 references paper 2 for some of its operating and performance characteristics, as it is primarily concerned with optical diagnostics.
As a result, I have included both papers and their data in the same directory. 

## Data description

There are two main types of data contained in these subfolders

1. Global performance and operating condition data, including discharge voltage, discharge current, mass flow rate, as well as thrust and background pressure, where available
2. LIF measurements of mean ion velocity
3. RPA measurements of ion flux

Performance/operational data is contained in each directory in a file called `performance.json`.
This contains a vector as a top-level object, with each element being one operating condition from the reference in that directory. 
The fields contained in each operating condition object are:

    - "background_pressure_Torr"
    - "anode_mass_flow_rate_mg_s"
    - "cathode_mass_flow_rate_mg_s"
    - "discharge_voltage_V"
    - "discharge_current_A"
    - "discharge_frequency_Hz"
    - "discharge_frequency_err_Hz"
    - "thrust_mN"

If a value is not available from a given reference, the value `"unknown"` will be provided instead of the listed value.

LIF data (axial coordinate, axial velocity, and uncertainty, if possible) are contained in CSV files prefixed with `LIF_SPT100`
