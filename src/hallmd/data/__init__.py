"""The `hallmd.data` package contains a folder for each unique thruster. The experimental data for each thruster
is further divided by folders for each individual paper or reference. The raw experimental data is contained within
these folders in any arbitrary format (h5, json, csv, etc.). Each set of raw experimental data should come with a
`dataloader.py` file that reads from the raw data into standardized Python objects. Any additional documentation
for the datasets is encouraged (e.g. citations, descriptions, summaries, etc.) and can be included in the data folders.

## Thrusters

### SPT-100
Currently the only thruster with available data. Data for the SPT-100 comes from three sources:

1. [Diamant et al. 2014](https://arc.aiaa.org/doi/10.2514/6.2014-3710) - provides thrust, cathode coupling voltage, and ion current density data as a function of chamber background pressure.
2. [Macdonald et al. 2019](https://arc.aiaa.org/doi/10.2514/1.B37133) - provides ion velocity profiles for varying chamber pressures.
3. [Sankovic et al. 1993](https://www.semanticscholar.org/paper/Performance-evaluation-of-the-Russian-SPT-100-at-Sankovic-Hamley/81b7d985669b21aa1a8419277c52e7a879bf3b46) - provides thrust at varying operating conditions.

Citations:
``` title="SPT-100.bib"
--8<-- "hallmd/data/spt100/spt100.bib:citations"
```
"""
# ruff: noqa: E501
