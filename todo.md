sketch out an OO design (keep data-parsing internal for now, but could be its own janus repo later)
  - mc, mcmc, sa, models, data, etc.
  - easy to version, deploy, test new PEMs

- python fit (finds directory)
- python fit single-fidelity (clears and plots)
- different executors (thread and process)
- check install and run on linux
- check setup_env.sh
- check pdm run_job
- check train_hpc.sh
- check slice scripts

- install from linux and windows
- test thruster 

- clean dependencies
- add H9 check
- upload to pypi
- update readme install, quickstart, project structure
- clean up docstrings and reference/theory docs

OO design
- Have a "Data" class that the hallmd testing framework knows how to work with
- Have a "Model" class similarly
- Have an "Analysis" class that knows how to combine "Model" and "Data"
- seems like we'll need to define a common file format for each qoi used by the model, and experiments should write a processing script that gets their raw data into that format -- should keep/record all raw data, processing scripts, notes on processing (README), as well as other metadata

Data details
- base Data - has units, name, measurement values, coords, metadata, uncertainty, operating conditions
- implementing Data classes (i.e. ThrusterData) would require a set of operating conditions
- Have a base "Dataloader" class that transforms raw file data into "Data" classes
- Would need a loader for each file type you support (and define requirements for what should be in those files)
- Could have h5, csv, txt, dat file loaders for example

- how would you handle "datasets", i.e. multiple qois for same operating conditions
- how would you handle the same qoi measured for different operating conditions (or even different metadata details)
- how would you handle time-resolved data
- how would you have a single file format for multiple qois or "Data" types
- would it be better so that people define a dataloader _per_ dataset (so they write a wrapper to load their own format into the common "Data") 
  - this would suck, better to have a single interface and people can translate their data to the format (so long as the interface is designed well)
- should you group "Data" by experiment/sweep/time or paper where it was recorded?

- what should a single "Data" be responsible for
- what should a single datafile be responsible for in the most generally useful way

- maybe assign a random id to "Data" when they get loaded
- Data with the same "name" can be concatenated (scalars at least)
- people won't want to specify a new file (with very duplicate metadata) for each data "point" in the same "experiment"
- most important for the "Model" is the operating conditions and the measurement/output itself (with its name, coords, and uncertainty)

Model would like:
- all QoIs grouped by operating condition (along with coords and uncertainty)
- so that model is evaluated once per condition and directly compared by output name to the measurements
- arrays of operating conditions for looping
- easy division into test/train sets (which involves grouping by experiment/datafile)

Experiments would like:
- QoIs grouped by operating condition (setup thruster -> run -> collect -> record)
- Grouped by experiment/datafile -- all files from the same day/test/campaign in one location

Questions:
- walk me through a typical experiment or test campaign (how long, how much data, etc.)
- how do you record data? proprietary binary formats or text/csv ? By-hand or automatic from logger?
- What operating conditions do you record/measure and their uncertainties
- What qois do you record/measure and their uncertainties?
- How do you represent scalar performance metrics v field/plasma properties with coords?
- How do you store and use the data later on?

Sean clark (UIUC)
- text files with collector thicknesses and carbon flux
- stl geometry files
- everything else (operating conditions) tracked by GT

Janice Cabrerra (GT)
- csv for everything (1d sweeps)
- only 5-ish operating conditions (3 pressures)
- no lif, mostly thrust, Id, jion 

Madison (UM)
- pressure, two ion gauges on a DAQ save every second or 100 ms, one on the wall, one a meter from the front of thruster and 1m behind, Phillips MKS (same as GT)
- thrust, depends on calibration and how to measure displacement, GT had coil current \propto displacement, UM uses optics to measure distance from a mirror, UM controls inclination (GT does not) -- should account for inclination calibrations
- faraday measures current density
- langmuir measures plasma potential (and cathode coupling voltage), and maybe emissive probe, various types of langmuir (cylinder, spherical for GIT), probe dimensions, UM using one from AFRL, might be useful to have the CAD
- ExB measures different populations (one design most people use), maybe currents through magnets
- LIF, ion velocity, wavelength dependent on gas type, doppler shift to get IVDF
- Incoherent Thompson scattering (ITS), same thing for EVDF
- LIF and ITS to get near-field plume properties using lasers, since probes mess with plasma, low signal-to-noise, amplifiers to increase signal-to-noise, UQ is hand-wavy
- RPA measures current density and IEDF (instead of faraday)
- faraday, langmuir, RPA on probe arm, different designs for all, different biasing schemes or number of grids, may bias collector plate
- recording _where_ measurements are taken (channel or thruster centerline), ExB is channel centerline, RPA are thruster centerline
- do measure oscillations for anode, cathode current, and cathode-gnd voltage, and discharge current
- Seth (CSU) Austen (?) doing time-resolved probe measurements
- thruster harnessing from power supply to thruster (length changes impedance, GT has long lines), artificially increase with an inductor to match GTs
- solenoids near thruster?
- also records mass flow rates, controllers are rated for different flow rates 10 sccm to 3000 sccm, gone back to 400sccm for Krypton
- can get uncertainty from the variance in the measurements
- oscillations measured via oscilloscope, but usually peak-to-peak measurements recorded by the daq and the RMS
- reaching thermal steady state (GT ~1 hour), thermocouples are often internal to the thruster in the bobin under the pole cover, should maybe record what "steady-state"
- thruster configuration (floating, cathode-tied, grounded)
- DAQ is labview, controls and monitors magnet currents, power supply, facility temperatures -- stores in excel file every couple seconds things for thruster, rows are each timestep, each column for qois like mass flow rate, currents, voltages, etc.
- laser diagnostics use csv/excel/txt, separate file for each location, record position in the filename, different tabs for laser temperature, power, etc. thruster itself, magnetic field settings, operating conditions, one wavelength per velocity, each column is some laser metadata like power, time avg window, etc.
- people usually do their own analysis for each probe depending on what is needed
- need to note velocity directions, different lasers needed for different direction