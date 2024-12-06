#=
Small wrapper for `HallThruster.run_simulation`.

Call as:
    `> julia run_hallthruster.jl infile.json --ref=main`
for a specific git ref (i.e. branch or commit hash),

Call as:
    `> julia run_hallthruster.jl infile.json --version=0.17.2`
for a specific version,

where `infile.json` is the input file. The output is saved at the location `infile['postprocess']['output_file']`.
=#
using Pkg

infile = ARGS[1]
ref_string = ARGS[2]

HALLTHRUSTER_URL = "https://github.com/UM-PEPL/HallThruster.jl"
HALLTHRUSTER_NAME = "HallThruster"

ref_type, ref = split(ref_string, "=")
is_version = occursin("version", ref_type)

# Find and activate the environment with the correct version of HallThruster.jl (or install it)
env_name = "hallthruster_$ref"
global_envs = joinpath(DEPOT_PATH[1], "environments")
env_path = joinpath(global_envs, env_name)

if isdir(env_path)
    Pkg.activate(env_path, io=devnull)

    # Update from git ref every time to get newest changes
    if !is_version
        Pkg.update(HALLTHRUSTER_NAME, io=devnull)
    end
else
    println("Creating environment $env_name...")
    mkpath(env_path)
    Pkg.activate(env_path)

    if is_version
        Pkg.add(name=HALLTHRUSTER_NAME, version=ref)  # should reuse from the global depot and make symlinks
    else
        Pkg.add(url=HALLTHRUSTER_URL, rev=ref)
    end

    println("Environment $env_name created successfully.")
end

@eval using HallThruster
HallThruster.run_simulation(infile)
