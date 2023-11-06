import json
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
from numpy.linalg.linalg import LinAlgError
from abc import ABC, abstractmethod
import logging
import sys
import uuid
import os

INPUT_DIR = Path(__file__).parent / 'input'
LOG_FORMATTER = logging.Formatter("%(asctime)s \u2014 [%(levelname)s] \u2014 %(name)-20s \u2014 %(message)s")


class ModelRunException(Exception):
    pass


class BaseRV(ABC):
    """Small wrapper class similar to scipy.stats random variables"""

    def __init__(self, id='', tex='', description='Random variable', units='-',
                 param_type='calibration', nominal=1, domain=(0, 1)):
        """Child classes must define sample/pdf functions"""
        self.bds = tuple(domain)
        self.nominal = nominal
        self.id = id if id != '' else str(uuid.uuid4())
        self.custom_id = id != ''
        self.tex = tex
        self.description = description
        self.units = units
        self.param_type = param_type    # One of (calibration, operating, design, simulation, coupling, other)

    def __repr__(self):
        return r'{}'.format(f"{self.id} - {self.description} ({self.units})")

    def __str__(self):
        return self.id

    def __eq__(self, other):
        """Consider two RVs equal if they share the same string representation (i.e. the same id), also returns true
        when checking if this RV is equal to a string id by itself"""
        return str(self) == str(other)

    def __hash__(self):
        return hash((self.id,))

    def to_tex(self, units=False, symbol=True):
        """Return a raw string that is well-formatted for plotting (with tex)
        :param units: whether to include the units in the string
        :param symbol: just latex symbol if true, otherwise the full description
        """
        s = self.tex if symbol else self.description
        if s == '':
            s = str(self)
        return r'{} [{}]'.format(s, self.units) if units else r'{}'.format(s)

    def bounds(self):
        """Return a tuple() of the domain of this RV"""
        return self.bds

    def update_bounds(self, lb, ub):
        self.bds = (lb, ub)

    def sample_domain(self, shape, method='random'):
        """Draw samples over the domain of this RV (not necessarily from the RV distribution)"""
        match method:
            case 'random':
                return np.random.rand(*shape) * (self.bds[1] - self.bds[0]) + self.bds[0]
            case other:
                raise NotImplementedError(f'Sampling method "{method}" is not known.')

    @abstractmethod
    def pdf(self, x):
        """Compute the PDF of the RV at x locations
        :param x: (...,) np.ndarray of any shape
        """
        pass

    @abstractmethod
    def sample(self, shape, nominal=None):
        """Draw samples from the RV
        :param shape: (...,) the shape of the returned samples
        :param nominal: an alternative nominal value to use (i.e. a center for relative Uniform or Normal)
        """
        pass


class ScalarRV(BaseRV):

    def __init__(self, **kwargs):
        """Implements a stand-in variable with no uncertainty/pdf, just scalars"""
        super().__init__(**kwargs)

    def pdf(self, x):
        return np.ones(x.shape)

    def sample(self, shape, **kwargs):
        return self.sample_domain(shape)


class UniformRV(BaseRV):

    def __init__(self, arg1, arg2, domain=None, **kwargs):
        """Implements a Uniform RV
        :param arg1: lower bound if specifying U(lb, ub), otherwise a tol or pct if specifying U(+/- tol/pct)
        :param arg2: upper bound if specifying U(lb, ub), otherwise a str() of either 'tol' or 'pct'
        """
        if isinstance(arg2, str):
            self.value = arg1   # Either an absolute tolerance or a relative percent
            self.type = arg2
        else:
            self.value = None
            self.type = 'bds'
        if self.type == 'bds':
            domain = (arg1, arg2) if domain is None else tuple(domain)     # This means domain overrides (arg1, arg2)
        else:
            domain = (0, 1) if domain is None else tuple(domain)
        super().__init__(domain=domain, **kwargs)

        # Set default nominal value as middle of the domain if not specified
        if kwargs.get('nominal', None) is None:
            self.nominal = (self.bds[1] + self.bds[0]) / 2

    # Override
    def __str__(self):
        return self.id if self.custom_id else f'U({self.bds[0]}, {self.bds[1]})'

    def get_uniform_bounds(self, nominal):
        """Return the correct set of bounds based on self.type"""
        match self.type:
            case 'bds':
                return self.bds  # Use the full preset domain
            case 'pct':
                if nominal is None:
                    return self.bds     # Default to full domain if nominal is not passed in
                return nominal * (1 - self.value), nominal * (1 + self.value)
            case 'tol':
                if nominal is None:
                    return self.bds     # Default to full domain if nominal is not passed in
                return nominal - self.value, nominal + self.value
            case other:
                raise NotImplementedError(f'self.type = {self.type} not known. Choose from ["pct, "tol", "bds"]')

    def pdf(self, x, nominal=None):
        bds = self.get_uniform_bounds(nominal)
        den = bds[1] - bds[0]
        den = 1 if np.isclose(den, 0) else den
        y = np.broadcast_to(1 / den, x.shape).copy()
        y[np.where(x > bds[1])] = 0
        y[np.where(x < bds[0])] = 0
        return y

    def sample(self, shape, nominal=None):
        bds = self.get_uniform_bounds(nominal)
        return np.random.rand(*shape) * (bds[1] - bds[0]) + bds[0]


class LogUniformRV(BaseRV):
    """Base 10 loguniform, only supports absolute bounds"""

    def __init__(self, log10_a, log10_b, **kwargs):
        super().__init__(**kwargs)
        self.bds = (10**log10_a, 10**log10_b)   # Probably the same as kwargs.get('domain')

    def __str__(self):
        return self.id if self.custom_id else f'LU({np.log10(self.bds[0]):.2f}, {np.log10(self.bds[1]):.2f})'

    def pdf(self, x):
        return np.log10(np.e) / (x * (np.log10(self.bds[1]) - np.log10(self.bds[0])))

    def sample(self, shape, nominal=None):
        lb = np.log10(self.bds[0])
        ub = np.log10(self.bds[1])
        return 10 ** (np.random.rand(*shape) * (ub - lb) + lb)


class LogNormalRV(BaseRV):
    """Base 10 lognormal"""

    def __init__(self, mu, std, domain=None, **kwargs):
        """Init with the mean and standard deviation of the underlying distribution, i.e. log10(x) ~ N(mu, std)"""
        if domain is None:
            domain = (10 ** (mu - 4*std), 10 ** (mu + 4*std))   # Use a default domain of +- 4std
        super().__init__(domain=domain, **kwargs)
        self.std = std
        self.mu = mu

    def recenter(self, mu, std):
        self.mu = mu
        self.std = std

    def __str__(self):
        return self.id if self.custom_id else f'LN_10({self.mu}, {self.std})'

    def pdf(self, x):
        return (np.log10(np.e) / (x * self.std * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((np.log10(x) - self.mu) / self.std) ** 2)

    def sample(self, shape, nominal=None):
        scale = np.log10(np.e)
        center = self.mu if nominal is None else nominal
        return np.random.lognormal(mean=(1 / scale) * center, sigma=(1 / scale) * self.std, size=shape)
        # return 10 ** (np.random.randn(*size)*self.std + center)  # Alternatively


class NormalRV(BaseRV):

    def __init__(self, mu, std, domain=None, **kwargs):
        if domain is None:
            domain = (mu - 2.5*std, mu + 2.5*std)   # Use a default domain of +- 2.5std
        super().__init__(domain=domain, **kwargs)
        self.mu = mu
        self.std = std

        # Set default nominal value as the provided mean
        if kwargs.get('nominal', None) is None:
            self.nominal = mu

    def recenter(self, mu, std=None):
        self.mu = mu
        if std is not None:
            self.std = std

    def __str__(self):
        return self.id if self.custom_id else f'N({self.mu}, {self.std})'

    def pdf(self, x):
        return (1 / (np.sqrt(2 * np.pi) * self.std)) * np.exp(-0.5 * ((x - self.mu) / self.std) ** 2)

    def sample(self, shape, nominal=None):
        center = self.mu if nominal is None else nominal
        return np.random.randn(*shape) * self.std + center


def load_variables(variables, file='variables.json'):
    """Load from variables.json into a list of BaseRV objects
    :param variables: list() of str() ids to match in json file
    :param file: json file in models/config to search for variable definitions
    :return vars: list() of corresponding BaseRV objects
    """
    with open(Path(__file__).parent / 'models' / 'config' / file, 'r') as fd:
        data = json.load(fd)

    vars = []
    keys = ['id', 'tex', 'description', 'units', 'param_type', 'nominal', 'domain']
    for str_id in variables:
        if str_id in data:
            var_info = data.get(str_id)
            kwargs = {key: var_info.get(key) for key in keys if var_info.get(key)}
            match var_info.get('rv_type', 'none'):
                case 'uniform_bds':
                    bds = var_info.get('rv_params')
                    vars.append(UniformRV(bds[0], bds[1], **kwargs))
                case 'uniform_pct':
                    vars.append(UniformRV(var_info.get('rv_params'), 'pct', **kwargs))
                case 'uniform_tol':
                    vars.append(UniformRV(var_info.get('rv_params'), 'tol', **kwargs))
                case 'normal':
                    mu, std = var_info.get('rv_params')
                    vars.append(NormalRV(mu, std, **kwargs))
                case 'none':
                    # Make a plain stand-in scalar RV object (no uncertainty)
                    vars.append(ScalarRV(**kwargs))
                case other:
                    raise NotImplementedError(f'RV type "{var_info.get("rv_type")}" is not known.')
        else:
            raise ValueError(f'You have requested the variable {str_id}, but it was not found in {file}. '
                             f'Please add a definition of {str_id} to {file} or construct it on your own.')

    return vars


def get_logger(name, stdout=True, log_file=None):
    """Setup a file/stdout logger with the given name"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    if stdout:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(std_handler)
    if log_file is not None:
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(f_handler)

    return logger


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


def spt100_data(qois=None):
    """Return a dict with experimental data for each specified quantity for the SPT-100
    :param qois: a list() of str specifying the experimental data to return, must be in ['V_cc', 'T', 'uion', 'jion']
    """
    if qois is None:
        qois = ['V_cc', 'T', 'uion', 'jion']
    exp_data = {qoi: None for qoi in qois}
    base_dir = Path(__file__).parent / 'data' / 'spt100'

    # Load Vcc data
    if 'V_cc' in exp_data:
        data = np.loadtxt(base_dir / 'vcc_dataset8.csv', delimiter=',', skiprows=1)
        x = data[:, [3, 1, 2]]  # (PB, Va, mdot_a)
        y = data[:, 4]
        noise_var = data[:, 5]
        exp_data['V_cc'] = dict(x=x, y=y, noise_var=noise_var)

    # Load thrust data
    if 'T' in exp_data:
        # files = [f for f in os.listdir(base_dir) if f.startswith('thrust')]
        files = ['thrust_dataset1.csv', 'thrust_dataset3.csv']
        x = np.zeros((0, 3))
        y = np.zeros((0, 1))
        noise_var = np.zeros((0, 1))
        for fname in files:
            thrust_data = np.loadtxt(base_dir / fname, delimiter=',', skiprows=1)
            x = np.concatenate((x, thrust_data[:, [3, 1, 2]]), axis=0)  # (PB, Va, mdot_a)
            y = np.concatenate((y, thrust_data[:, [4]]), axis=0)
            noise_var = np.concatenate((noise_var, thrust_data[:, [5]]), axis=0)
        y = np.squeeze(y) * 1e-3                    # N
        noise_var = np.squeeze(noise_var) * 1e-6    # N^2
        exp_data['T'] = dict(x=x, y=y, noise_var=noise_var)

    # Load ion velocity data
    if 'uion' in exp_data:
        # files = [f for f in os.listdir(base_dir) if f.startswith('ui')]
        files = ['ui_dataset5.csv']
        x = np.zeros((0, 4))
        y = np.zeros((0, 1))
        noise_var = np.zeros((0, 1))
        for fname in files:
            data = np.loadtxt(base_dir / fname, delimiter=',', skiprows=1)
            x = np.concatenate((x, data[:, [3, 1, 2, 4]]), axis=0)  # (PB, Va, mdot_a, z)
            y = np.concatenate((y, data[:, [5]]), axis=0)
            noise_var = np.concatenate((noise_var, data[:, [6]]), axis=0)
        y = np.squeeze(y) * 1e-3  # km/s
        noise_var = np.squeeze(noise_var) * 1e-6  # (km/s)^2
        exp_data['uion'] = dict(x=x, y=y, noise_var=noise_var)

    # Load ion velocity data
    if 'jion' in exp_data:
        files = [f for f in os.listdir(base_dir) if f.startswith('jion')]
        x = np.zeros((0, 5))
        y = np.zeros((0, 1))
        noise_var = np.zeros((0, 1))
        for fname in files:
            data = np.loadtxt(base_dir / fname, delimiter=',', skiprows=1)
            x = np.concatenate((x, data[:, [3, 1, 2, 4, 5]]), axis=0)  # (PB, Va, mdot_a, r_m, alpha_deg)
            y = np.concatenate((y, data[:, [6]]), axis=0)
            noise_var = np.concatenate((noise_var, data[:, [7]]), axis=0)
        y = np.squeeze(y) * 10  # A/m^2
        noise_var = np.squeeze(noise_var) * 10**2  # (A/m^2)^2
        exp_data['jion'] = dict(x=x, y=y, noise_var=noise_var)

    return exp_data
