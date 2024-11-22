#=
Small wrapper for `HallThruster.run_simulation`. Call as:

    `> julia run_hallthruster.jl infile.json`

where `infile.json` is the input file. The output is saved at the location `infile['postprocess']['output_file']`.
=#

using HallThruster
infile = ARGS[1]
HallThruster.run_simulation(infile)
println("Hello world!")
