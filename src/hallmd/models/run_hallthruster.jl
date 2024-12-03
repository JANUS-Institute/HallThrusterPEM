#=
Small wrapper for `HallThruster.run_simulation`. Call as:

    `> julia run_hallthruster.jl infile.json version`

where `infile.json` is the input file and `version` is the HallThruster.jl version. The output is saved
at the location `infile['postprocess']['output_file']`.
=#
using Pkg
infile = ARGS[1]
version = ARGS[2]

# Find and activate the environment with the correct version of HallThruster.jl (or install it)
env_name = "hallthruster_$version"
global_envs = joinpath(homedir(), ".julia", "environments")
env_path = joinpath(global_envs, env_name)

if isdir(env_path)
    Pkg.activate(env_path)
else
    println("Creating environment $env_name...")
    mkpath(env_path)
    Pkg.activate(env_path)
    Pkg.add(name="HallThruster", version=version)
    println("Environment $env_name created successfully.")
end

@eval using HallThruster
HallThruster.run_simulation(infile)
