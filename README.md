# HallThrusterPEM
Prototype of a predictive engineering model (PEM) of a Hall thruster, integrating physics-based sub-models with uncertainty quantification.

## Project structure
```
HallThrusterPEM                 # Root project directory
|- data                         # Store experimental data in .csv file format
|  |- spt100                    # Data for the SPT-100
|  |- ...
|- doc                          # Any relevant documentation or references
|- input                        # Input .json files for each sub-model
|  |- thruster_input.json
|  |- ...
|- models                       # Python wrappers for sub-models
|  |- thruster.py
|  |- ...
|- results                      # Test scripts write data to this directory (but kept out of the repo)
|- sandbox                      # Staging area for testing/writing one-off scripts and functions
|- test                         # Python scripts for testing models, generating data, and plotting results
|- utils.py                     # Useful utility functions
```
