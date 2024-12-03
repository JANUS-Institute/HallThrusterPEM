#=
Small script to simulate `HallThruster.run_simulation`. Call as:

    `> julia sim_hallthruster.jl infile.json`

where `infile.json` is the input file. The output is saved at the location `infile['postprocess']['output_file']`.
=#
using JSON

function read_json(file_path::String)
    open(file_path, "r") do f
        return JSON.parse(read(f, String))
    end
end

function write_json(file_path::String, data::Dict)
    open(file_path, "w") do f
        JSON.print(f, data, 4)
    end
end

function perform_operations(input_path::String)
    json_data = read_json(input_path)

    # Access some inputs for demonstration
    discharge_voltage = json_data["config"]["discharge_voltage"]
    cathode_potential = json_data["config"]["cathode_potential"]
    flow_rate = json_data["config"]["anode_mass_flow_rate"]
    anom_coeff = json_data["config"]["anom_model"]["model"]["c1"]
    ncells = json_data["simulation"]["ncells"]
    domain = json_data["config"]["domain"]

    output_path = json_data["postprocess"]["output_file"]

    # Simulate calculation of average output QoIs
    q = 1.6e-19
    m_ion = 2.18e-25
    beam_current = (q / m_ion) * flow_rate  # A
    current_eff = (1 - anom_coeff * 2)
    discharge_current = beam_current / current_eff
    v_exh = sqrt(2 * q * (discharge_voltage - cathode_potential) / m_ion)
    thrust = flow_rate * v_exh
    mass_eff = 1 - anom_coeff * 5
    voltage_eff = 1 - anom_coeff * 2
    anode_eff = 0.5 * thrust^2 / (flow_rate * discharge_voltage * discharge_current)

    z = range(domain[1], stop=domain[2], length=ncells)
    uion = v_exh ./ (1 .+ exp.(-100 .* (z .- 0.04)))

    # Create the output JSON object
    output_json = Dict("outputs" => Dict("average" => Dict(
    "thrust" => thrust,
    "ion_current" => beam_current,
    "current_eff" => current_eff,
    "discharge_current" => discharge_current,
    "v_exh" => v_exh,
    "mass_eff" => mass_eff,
    "voltage_eff" => voltage_eff,
    "anode_eff" => anode_eff,
    "ui_1" => uion,
    "z" => z
    )), "config" => json_data["config"],
    "simulation" => json_data["simulation"],
    "postprocess" => json_data["postprocess"])

    write_json(output_path, output_json)

end

# Perform the operations
perform_operations(ARGS[1])
