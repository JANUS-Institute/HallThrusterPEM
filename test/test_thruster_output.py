# Standard imports
import json
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
logging.basicConfig(level=logging.INFO)

# Custom imports
from models.plume import current_density_model as plume_model
from utils import data_load, data_write


def main():
    N = [50, 50]
    loc = [np.linspace(0.001, 0.01, N[0]), np.linspace(-90, 90, N[1])]
    pt_grids = np.meshgrid(*loc)
    x_loc = np.vstack([grid.ravel() for grid in pt_grids]).T  # (np.prod(Nx), x_dim)
    r_vec = x_loc[:, 0]
    alpha_vec = x_loc[:, 1]

    # Right r and alpha to plume input file
    data = data_load('plume_input.json')
    data['other']['r'] = list(r_vec)
    data['other']['alpha'] = list(alpha_vec)
    data_write(data, 'plume_input.json')

    # Run plume model
    j_ion = plume_model(plume_input='plume_input.json', thruster_input='thruster_output.json')

    # Reform grids
    r_grid, alpha_grid = [x_loc[:, i].reshape((N[1], N[0])) for i in range(2)]  # reform grids
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
