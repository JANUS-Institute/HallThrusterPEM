# Standard imports
import matplotlib.pyplot as plt
import sys
import numpy as np
import logging

sys.path.append('..')
logging.basicConfig(level=logging.INFO)

# Custom imports
from models.cc import cathode_coupling_model as cc_model
from models.thruster import hall_thruster_jl_model as thruster_model
from models.plume import current_density_model as plume_model
from utils import data_load, data_write


# TODO: More consistent way to input parameters common to multiple models (PB, Te, geometry, etc.)

def main():
    # Run cathode model
    cc_model(cc_input='cc_input.json')

    # Run thruster model
    thruster_model(thruster_input='thruster_input.json', cc_input='cc_output.json')  # produces thruster_output.json

    # Run plume model on thruster output
    j_ion = plume_model(plume_input='plume_input.json', thruster_input='thruster_output.json')

    # Reform grids
    N = [50, 50]
    input_data = data_load('plume_input.json')
    r = np.array(input_data['other']['r'])
    alpha = np.array(input_data['other']['alpha'])
    r_grid, alpha_grid = [r.reshape((N[1], N[0])), alpha.reshape((N[1], N[0]))]
    x_grid = r_grid * np.cos(alpha_grid * np.pi / 180)
    y_grid = r_grid * np.sin(alpha_grid * np.pi / 180)
    j_ion_grid = j_ion.reshape((N[1], N[0]))

    # Plot results
    plt.figure()
    c = plt.contourf(x_grid, y_grid, j_ion_grid, 60, cmap='jet')
    cbar = plt.colorbar(c)
    cbar.set_label(r'Ion current density ($A/m^2$)')
    plt.xlabel(r'Distance from thruster exit [m]')
    plt.ylabel(r'Distance from channel centerline [m]')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
