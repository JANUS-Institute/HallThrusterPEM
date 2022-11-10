# Module for thruster models
import logging
import sys
from pathlib import Path
import math
import json
import tempfile
import os
import juliacall

Q_E = 1.602176634e-19   # Fundamental charge (C)
sys.path.append('..')
logger = logging.getLogger(__name__)

from utils import ModelRunException


def hallthruster_jl_input(thruster_input):
    # Format inputs for Hallthruster.jl
    json_data = dict()
    data_dir = Path('../data')
    json_data['parameters'] = {'neutral_temp_K': thruster_input['neutral_temp_K'],
                               'neutral_velocity_m_s': thruster_input['neutral_velocity_m_s'],
                               'ion_temp_K': thruster_input['ion_temp_K'],
                               'cathode_electron_temp_eV': thruster_input['cathode_electron_temp_eV'],
                               'sheath_loss_coefficient': thruster_input['sheath_loss_coefficient'],
                               'inner_outer_transition_length_m': thruster_input['inner_outer_transition_length_m'],
                               'anom_model_coeffs': [thruster_input['anom_coeff_1'], thruster_input['anom_coeff_2']],
                               'background_pressure_Torr': thruster_input['background_pressure_Torr'],
                               'background_temperature_K': thruster_input['background_temperature_K'],
                               }
    json_data['design'] = {'thruster_name': thruster_input['thruster_name'],
                           'inner_radius': thruster_input['inner_radius'],
                           'outer_radius': thruster_input['outer_radius'],
                           'channel_length': thruster_input['channel_length'],
                           'magnetic_field_file': str(data_dir / thruster_input['magnetic_field_file']),
                           'wall_material': thruster_input['wall_material'],
                           'magnetically_shielded': thruster_input['magnetically_shielded'],
                           'anode_potential': thruster_input['anode_potential'],
                           'cathode_potential': thruster_input['cathode_potential'],
                           'anode_mass_flow_rate': thruster_input['anode_mass_flow_rate'],
                           'propellant': thruster_input['propellant_material'],
                           }
    json_data['simulation'] = {'num_cells': thruster_input['num_cells'],
                               'dt_s': thruster_input['dt_s'],
                               'duration_s': thruster_input['duration_s'],
                               'num_save': thruster_input['num_save'],
                               'cathode_location_m': thruster_input['cathode_location_m'],
                               'max_charge': thruster_input['max_charge'],
                               'flux_function': thruster_input['flux_function'],
                               'limiter': thruster_input['limiter'],
                               'reconstruct': thruster_input['reconstruct'],
                               'ion_wall_losses': thruster_input['ion_wall_losses'],
                               'electron_ion_collisions': thruster_input['electron_ion_collisions'],
                               'anom_model': thruster_input['anom_model'],
                               'solve_background_neutrals': thruster_input['solve_background_neutrals']
                               }

    return json_data


def hall_thruster_jl_model(thruster_input, jl=None):
    # Import Julia
    if jl is None:
        from juliacall import Main as jl
        jl.seval("using HallThruster")

    # Format inputs for Hallthruster.jl
    json_data = hallthruster_jl_input(thruster_input)

    # Run simulation
    try:
        fd = tempfile.NamedTemporaryFile(suffix='.json', encoding='utf-8', mode='w', delete=False)
        json.dump(json_data, fd, ensure_ascii=False, indent=4)
        fd.close()
        sol = jl.HallThruster.run_simulation(fd.name)
        os.unlink(fd.name)   # delete the tempfile
    except juliacall.JuliaError as e:
        raise ModelRunException(f"Julicall error in Hallthruster.jl: {e}")

    if str(sol.retcode).lower() != "success":
        raise ModelRunException(f"Exception in Hallthruster.jl: Retcode = {sol.retcode}")

    # Load simulation results
    fd = tempfile.NamedTemporaryFile(suffix='.json', encoding='utf-8', mode='w', delete=False)
    fd.close()
    jl.HallThruster.write_to_json(fd.name, jl.HallThruster.time_average(sol))
    with open(fd.name, 'r') as f:
        thruster_output = json.load(f)
    os.unlink(fd.name)  # delete the tempfile

    j_exit = 0      # Current density at thruster exit
    ui_exit = 0     # Ion velocity at thruster exit
    for param, grid_sol in thruster_output[0].items():
        if 'niui' in param:
            charge_num = int(param.split('_')[1])
            j_exit += Q_E * charge_num * grid_sol[-1]
        if param.split('_')[0] == 'ui':
            ui_exit += grid_sol[-1]

    A = math.pi * (thruster_input['outer_radius'] ** 2 - thruster_input['inner_radius'] ** 2)
    ui_avg = ui_exit / thruster_input['max_charge']
    I_B0 = j_exit * A           # Total current (A) at thruster exit

    thruster_output[0].update({'avg_ion_velocity': ui_avg, 'I_B0': I_B0})

    return thruster_output[0]
