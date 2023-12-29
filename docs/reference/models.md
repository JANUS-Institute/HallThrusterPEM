All models are specified in `hallmd.models`. Currently supported models are based on a three-component feedforward
system for a Hall thruster:

1. **Cathode** - Accounts for interactions of the cathode plasma with the main discharge.
2. **Thruster** - The primary simulation of the Hall thruster channel discharge and near-field.
3. **Plume** - Models the far-field expansion of the plasma plume in the vacuum chamber.

Individual component models are integrated into a predictive engineering model (PEM) multidisciplinary system in
`hallmd.models.pem`.

## Model configuration
Component models can optionally retrieve external configuration data from the `hallmd.models.config` directory. For 
example, `Hallthruster.jl` obtains magnetic field information from `bfield_spt100.csv` stored in the `config` directory, 
as well as additional configurations from `hallthruster_jl.json`.