# Module for thruster models

from juliacall import Main as jl
import logging
import sys
from pathlib import Path
import math

Q_E = 1.602176634e-19   # Fundamental charge (C)
sys.path.append('..')
logger = logging.getLogger(__name__)
jl.seval("using HallThruster")

from utils import data_load, data_write


def hall_thruster_jl_model(thruster_input):
    logger.info('Running thruster model')
    data_dir = Path('../data')
    interface_dir = Path('../interface')

    # Create json input file for julia model
    json_data = dict()
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
    data_write(json_data, 'hallthruster_jl_in.json')

    # Call Hallthruster.jl simulation
    sol = jl.HallThruster.run_simulation(str(interface_dir / 'hallthruster_jl_in.json'))
    jl.HallThruster.write_to_json(str(interface_dir / 'hallthruster_jl_out.json'), jl.HallThruster.time_average(sol))

    # Return quantities of interest
    thruster_output = data_load('hallthruster_jl_out.json')
    j_exit = 0      # Current density at thruster exit
    ui_exit = 0     # Ion velocity at thruster exit
    n_avg = 0
    for param, grid_sol in thruster_output[0].items():
        if 'niui' in param:
            charge_num = int(param.split('_')[1])
            j_exit += Q_E * charge_num * grid_sol[-1]
        if param.split('_')[0] == 'ui':
            ui_exit += grid_sol[-1]
            n_avg += 1

    A = math.pi * (thruster_input['outer_radius'] ** 2 - thruster_input['inner_radius'] ** 2)
    ui_avg = ui_exit / n_avg
    I_B0 = j_exit * A           # Total current (A) at thruster exit

    return ui_avg, I_B0
