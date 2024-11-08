decide on scripts/src desired usage
update amisc things
sketch out an OO design (keep data-parsing internal for now, but could be its own janus repo later)
  - mc, mcmc, sa, models, data, etc.
  - easy to version, deploy, test new PEMs

- clean dependencies
- add H9 check
- upload to pypi
- update readme install, quickstart, project structure
- clean up docstrings and reference/theory docs

OO design
- Have a "Data" class that the hallmd testing framework knows how to work with
- Have a "Model" class similarly
- Have an "Analysis" class that knows how to combine "Model" and "Data"

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