Thruster model:
- thruster_config, simulation_config, and postprocess_config and how they correspond to HallThruster.jl Thruster, Config, and post-processing steps
- Format of thruster device.yml configuration (essentially just match hallthruster.jl Thruster for now)
- PEM to JULIA conversion
    - essentially allows hallthruster_jl wrapper function to be used for any set of inputs to Hallthruster.jl, as long as you specify how to map the pem name to the location in the julia input struct
    - Can also specify a static function to convert fidelity indices to configurations within Hallthruster.jl (i.e. (2, 2) -> ncharge, dt, ncells))
    - Only things not supported flexibly like this are special cases where a julia input is not a simple lookup/replacement -- i.e. like c2 = a1 * a2. For those cases, just add the handling logic directly in _fmt_input(). But might in the future allow some kind of user-specifiable function.
    - outputs returned by the wrapper must be located in "outputs" in the pem-julia mapping dict

PEM:
- Usage of scripts folder
- PEM amisc config files and referencing hallmd.models