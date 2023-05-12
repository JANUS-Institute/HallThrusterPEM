import json
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
from numpy.linalg.linalg import LinAlgError
from scipy.stats import uniform, norm
import os

INPUT_DIR = Path(__file__).parent / 'input'


class ModelRunException(Exception):
    pass


class BaseRV:
    """Small wrapper class for scipy.stats random variables"""

    def __init__(self):
        """Child classes must define bounds and a scipy.stats distribution"""
        self.bds = None     # (tuple)
        self.rv = None      # Frozen distribution from scipy.stats
        self.disp = ''      # Display name for printing

    def __repr__(self):
        return self.disp

    def __str__(self):
        return self.__repr__()

    def bounds(self):
        """Return the bounds of the RV"""
        return self.bds

    def pdf(self, x):
        """Compute the PDF of the RV at x locations
        :param x: (...,) np.ndarray of any shape
        """
        return self.rv.pdf(x)

    def sample(self, shape):
        """Draw samples from the RV
        :param shape: (...,) the shape of the returned samples
        """
        return self.rv.rvs(size=shape)


class UniformRV(BaseRV):

    def __init__(self, a, b, name=''):
        super().__init__()
        self.disp = f'U({a}, {b})' if name == '' else name
        self.bds = (a, b)
        self.rv = uniform(loc=a, scale=b-a)

    def update_bounds(self, lb, ub):
        self.bds = (lb, ub)
        self.rv = uniform(loc=lb, scale=ub-lb)


class NormalRV(BaseRV):

    def __init__(self, mean, std, name=''):
        super().__init__()
        self.disp = f'N({mean}, {std})' if name == '' else name
        self.bds = (mean - 4*std, mean + 4*std)
        self.rv = norm(loc=mean, scale=std)


def print_stats(data, logger=None):
    """Print stats of 1D data"""
    log_func = print if logger is None else logger.info
    log_func(f"{'Average': >8} {'Std dev': >8} {'Minimum': >8} {'25 pct': >8} {'50 pct': >8} {'75 pct': >8} "
             f"{'Maximum': >8}")
    log_func(f"{np.mean(data): 8.1f} {np.sqrt(np.var(data)): 8.1f} {np.min(data): 8.1f} "
             f"{np.percentile(data, 25): 8.1f} {np.percentile(data, 50): 8.1f} {np.percentile(data, 75): 8.1f} "
             f"{np.max(data): 8.1f}")


def ax_default(ax, xlabel='', ylabel='', legend=True):
    """Nice default formatting for plotting X-Y data"""
    plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    if legend:
        leg = ax.legend(fancybox=True)
        frame = leg.get_frame()
        frame.set_edgecolor('k')


def get_cycle(cmap, N=None, use_index="auto"):
    """Get a color cycler for plotting"""
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index = True
            else:
                use_index = False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index == "auto":
        if cmap.N > 100:
            use_index = False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0, 1, N))
        return cycler("color", colors)


def data_load(filename, dir=INPUT_DIR):
    """Convenience function to load .json data files"""
    with open(Path(dir) / filename, 'r') as fd:
        data = json.load(fd)

    if 'input' in filename and 'system' not in filename:
        load_system_inputs(data)

    return data


def data_write(data, filename, dir=INPUT_DIR):
    """Convenience function to write .json data files"""
    with open(Path(dir) / filename, 'w', encoding='utf-8') as fd:
        json.dump(data, fd, ensure_ascii=False, indent=4)


def load_system_inputs(input_data):
    """Overwrite input parameters in 'input_data' with system input values"""
    system_data = data_load('system_input.json')
    for input_type, input_dict in system_data.items():
        for input_param, sys_value in input_dict.items():
            if input_data[input_type].get(input_param):
                input_data[input_type][input_param] = copy.deepcopy(sys_value)


def set_model_inputs(model_name, inputs_to_set):
    """Set values for model inputs"""
    model_data = data_load(f'{model_name}_input.json')
    for input_type, input_dict in model_data.items():
        for input_param in input_dict:
            if input_param in inputs_to_set:
                model_data[input_type][input_param]['nominal'] = inputs_to_set[input_param]
    data_write(model_data, f'{model_name}_input.json')


def parse_input_file(file, exclude=None):
    """Parse generic component input file, exclude uncertainty for input types optional"""
    if exclude is None:
        exclude = []

    nominal_input = {}
    input_uncertainty = {}
    if file == 'thruster_input.json':
        nominal_input, input_uncertainty = parse_thruster_input(file, exclude=exclude)
    else:
        input_data = data_load(file)
        for input_type, input_params in input_data.items():
            for param, param_value in input_params.items():
                nominal_input[param] = param_value['nominal']
                if param_value['uncertainty'] != 'none' and input_type not in exclude:
                    input_uncertainty[param] = {'uncertainty': param_value['uncertainty'],
                                                'value': param_value['value']}

    return nominal_input, input_uncertainty


def parse_thruster_input(file, exclude=None):
    """Helper function to parse thruster input"""
    if exclude is None:
        exclude = []

    thruster_data = data_load(file)
    thruster_input = {}
    input_uncertainty = {}

    # Loop over all inputs and parse into thruster_input and input_uncertainty
    for input_type, input_params in thruster_data.items():
        for param, param_value in input_params.items():
            if input_type == 'simulation':
                thruster_input[param] = param_value
            elif 'material' in param:
                # Handle specifying material properties
                for mat_prop, prop_value in param_value.items():
                    if mat_prop == 'name':
                        thruster_input[param] = prop_value
                    else:
                        if prop_value['uncertainty'] != 'none' and input_type not in exclude:
                            input_uncertainty[mat_prop] = {'uncertainty': prop_value['uncertainty'],
                                                           'value': prop_value['value']}
            elif param == 'magnetic_field':
                # Handle different ways to specify magnetic field profile
                thruster_input['magnetic_field_file'] = param_value['magnetic_field_file']
                if param_value['uncertainty'] != 'none' and input_type not in exclude:
                    input_uncertainty[param] = {'uncertainty': param_value['uncertainty'],
                                                'value': param_value['value']}
            else:
                # Handle all generic input parameters
                thruster_input[param] = param_value['nominal']
                if param_value['uncertainty'] != 'none' and input_type not in exclude:
                    input_uncertainty[param] = {'uncertainty': param_value['uncertainty'],
                                                'value': param_value['value']}
    return thruster_input, input_uncertainty


def approx_hess(func, theta, pert=0.01):
    """Approximate Hessian of the function at a specified theta location
    Parameters
    ----------
    func: expects to be called as func(theta) -> (*, y_dim)
    theta: (*, theta_dim) point to linearize model about
    pert: Perturbation for approximate partial derivatives

    Returns
    -------
    H: (*, theta_dim, theta_dim) The approximate Hessian (theta_dim, theta_dim) at locations (*)
    """
    theta = np.atleast_1d(theta)
    shape = theta.shape[:-1]                # (*)
    theta_dim = theta.shape[-1]             # Number of parameters
    dtheta = pert * theta

    # Return a Hessian (theta_dim, theta_dim) at locations (*)
    H = np.zeros((*(shape[:-1]), theta_dim, theta_dim))

    for i in range(theta_dim):
        for j in range(i, theta_dim):
            # Allocate space at 4 grid points (n1=-1, p1=+1)
            theta_n1_n1 = np.copy(theta)
            theta_p1_p1 = np.copy(theta)
            theta_n1_p1 = np.copy(theta)
            theta_p1_n1 = np.copy(theta)

            # Perturbations to theta in each direction
            theta_n1_n1[..., i] -= dtheta[..., i]
            theta_n1_n1[..., j] -= dtheta[..., j]
            f_n1_n1 = func(theta_n1_n1)

            theta_p1_p1[..., i] += dtheta[..., i]
            theta_p1_p1[..., j] += dtheta[..., j]
            f_p1_p1 = func(theta_p1_p1)

            theta_n1_p1[..., i] -= dtheta[..., i]
            theta_n1_p1[..., j] += dtheta[..., j]
            f_n1_p1 = func(theta_n1_p1)

            theta_p1_n1[..., i] += dtheta[..., i]
            theta_p1_n1[..., j] -= dtheta[..., j]
            f_p1_n1 = func(theta_p1_n1)

            res = (f_n1_n1 + f_p1_p1 - f_n1_p1 - f_p1_n1) / np.expand_dims(4 * dtheta[..., i] * dtheta[..., j],
                                                                           axis=-1)

            # Hessian only computed for scalar functions, y_dim=1 on last axis
            H[..., i, j] = np.squeeze(res, axis=-1)
            H[..., j, i] = np.squeeze(res, axis=-1)

    return H


def batch_normal_sample(mean, cov, size: "tuple | int" = ()):
    """
    Batch sample multivariate normal distributions.
    https://stackoverflow.com/questions/69399035/is-there-a-way-of-batch-sampling-from-numpys-multivariate-normal-distribution-i
    Arguments:
        mean: expected values of shape (…M, D)
        cov: covariance matrices of shape (…M, D, D)
        size: additional batch shape (…B)
    Returns: samples from the multivariate normal distributions
             shape: (…B, …M, D)
    """
    # Make some checks on input
    mean = np.atleast_1d(mean)
    cov = np.atleast_1d(cov)
    dim = cov.shape[0]

    # 1-D case
    if dim == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(mean.shape) == 1:
        mean = mean[np.newaxis, :]

    assert cov.shape[0] == cov.shape[1] == dim
    assert mean.shape[-1] == dim

    size = (size, ) if isinstance(size, int) else tuple(size)
    shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
    X = np.random.standard_normal((*shape, 1)).astype(np.float32)
    L = np.linalg.cholesky(cov)
    return (L @ X).reshape(shape) + mean


def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except LinAlgError:
        return False


def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def spt100_data(exclude=None):
    """Return a list of tuples of the form (data_type, operating conditions, data, experimental noise)"""
    exp_data = []
    base_dir = Path(__file__).parent / 'data' / 'spt100'

    if exclude is None:
        exclude = []

    # Load Vcc data
    if 'vcc' not in exclude:
        data = np.loadtxt(base_dir / 'vcc_dataset8.csv', delimiter=',', skiprows=1)
        for i in range(data.shape[0]):
            x = {'anode_potential': data[i, 1],
                 'anode_mass_flow_rate': data[i, 2]*1e-6,
                 'background_pressure_Torr': data[i, 3]}
            y = data[i, 4]
            var = {'noise_var': 'constant', 'value': (0.3/2)**2}
            exp_data.append(('cc_voltage', x.copy(), y, var.copy()))

    # Load thrust data
    if 'thrust' not in exclude:
        # files = [f for f in os.listdir(base_dir) if f.startswith('thrust')]
        files = ['thrust_dataset1.csv', 'thrust_dataset3.csv']
        data = np.zeros((1, 6))
        for fname in files:
            thrust_data = np.loadtxt(base_dir / fname, delimiter=',', skiprows=1)
            thrust_data = np.atleast_2d(thrust_data).reshape((-1, 6))
            data = np.concatenate((data, thrust_data), axis=0)
        data = data[1:, :]

        for i in range(data.shape[0]):
            x = {'anode_potential': data[i, 1],
                 'anode_mass_flow_rate': data[i, 2] * 1e-6,
                 'background_pressure_Torr': data[i, 3]}
            y = data[i, 4] * 1e-3  # N
            var = {'noise_var': 'percent', 'value': 0.9}
            exp_data.append(('thrust', x.copy(), y, var.copy()))

    # Load ion velocity data
    if 'ion_velocity' not in exclude:
        data = np.loadtxt(base_dir / 'ui_dataset5.csv', delimiter=',', skiprows=1)
        Np = 3
        data = data.reshape((Np, -1, 7))
        for i in range(Np):
            x = {'anode_potential': data[i, 0, 1],
                 'anode_mass_flow_rate': data[i, 0, 2] * 1e-6,
                 'background_pressure_Torr': data[i, 0, 3],
                 'z_m': list(data[i, :, 4])}
            y = list(data[i, :, 5])
            var = {'noise_var': 'constant', 'value': 62500}
            exp_data.append(('axial_ion_velocity', copy.deepcopy(x), y.copy(), var.copy()))

        # data = np.loadtxt(base_dir / 'ui_dataset6.csv', delimiter=',', skiprows=1)
        # x = {'anode_potential': data[0, 1],
        #      'anode_mass_flow_rate': data[0, 2] * 1e-6,
        #      'background_pressure_Torr': data[0, 3],
        #      'z_m': list(data[:, 4])}
        # y = list(data[:, 5])
        # var = {'noise_var': 'constant', 'value': 62500}
        # exp_data.append(('axial_ion_velocity', copy.deepcopy(x), y.copy(), var.copy()))

    # Load ion current density data
    if 'ion_current_density' not in exclude:
        data = np.loadtxt(base_dir / 'jion_dataset4.csv', delimiter=',', skiprows=1)
        Np = 8
        data = data.reshape((Np, -1, 8))
        for i in range(Np):
            x = {'anode_potential': data[i, 0, 1],
                 'anode_mass_flow_rate': data[i, 0, 2] * 1e-6,
                 'background_pressure_Torr': data[i, 0, 3],
                 'r_m': list(data[i, :, 4]),
                 'alpha_deg': list(data[i, :, 5])}
            y = list(data[i, :, 6] * 10)  # A/m^2
            var = {'noise_var': 'percent', 'value': 20}
            exp_data.append(('ion_current_density', copy.deepcopy(x), y.copy(), var.copy()))

    return exp_data
