The `hallmd.data` package contains a folder for each unique thruster. The experimental data for each thruster is
further divided by folders for each individual paper or reference. The raw experimental data is contained within these 
folders in any arbitrary format (hdf4, json, csv, etc.). Each set of raw experimental data should come with a 
`dataloader.py` file that reads from the raw data into standardized Python objects. Any additional documentation for
the datasets is encouraged (e.g. citations, descriptions, summaries, etc.) and can be included in the data folders.

## Thrusters
- **SPT-100** - currently the only available thruster.

## Top-level `loader.py`
This module provides high-level convenience functions for loading data for specific thrusters. If you plan to add
experimental data for a new thruster, this would be a good location for a wrapper function to load this data.

::: hallmd.data.loader