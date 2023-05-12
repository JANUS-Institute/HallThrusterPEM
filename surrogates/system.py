"""Module for system-level adaptive multidisciplinary, multi-fidelity surrogate implementation"""
import numpy as np
import networkx as nx
import itertools
import sys
import os
import time
import datetime
import functools
import logging
import dill
from abc import ABC, abstractmethod
from datetime import timezone
from pathlib import Path

sys.path.append('..')

from utils import UniformRV

# Setup logging for the module
FORMATTER = logging.Formatter("%(asctime)s \u2014 [%(levelname)s] \u2014 %(name)-36s \u2014 %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(FORMATTER)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class SystemSurrogate:

    def __init__(self, components, adj_matrix, exo_vars, coupling_bds, est_bds=0, log_dir=None, save_dir='.'):
        """Construct a multidisciplinary system surrogate
        :param components: list(Nk,) of dicts specifying component name (str), a callable function model(x, alpha), a
                           string specifying surrogate class type, the highest 'truth' set of model fidelity indices,
                           the maximum level of model fidelity indices, the maximum surrogate refinement level, the
                           global indices of required system exogenous inputs, a dict specifying the indices of local
                           coupling variable inputs from other models, and a list() that maps local component outputs
                           to global coupling indices
                           Ex: [{'name': 'Thruster', 'model': callable(x, alpha), 'truth_alpha': (3,), 'max_alpha': (3),
                                         'max_beta': 5, 'exo_in': [0, 2, 6...], 'local_in': {'model1': [1,2,3],
                                         'model2': [0]}, 'global_out': [0, 1, 2, 3]},
                                {'name': 'Plume', ...}, ...]
        :param adj_matrix: (Nk, Nk), matrix of 0,1 specifying directed edges from->to for the Nk components
        :param exo_vars: list() of BaseRVs specifying the bounds/distributions for all system-level exogenous inputs
        :param coupling_bds: list() of tuples specifying estimated bounds for all coupling variables
        :param est_bds: (int) number of samples to estimate coupling variable bounds, do nothing if 0
        :param log_dir: (str) write log files to this directory if specified
        :param save_dir: (str) directory to save .pkl files during refinement containing the SystemSurrogate object
        """
        # Number of components
        from surrogates.polynomial import LagrangeSurrogate
        Nk = len(components)
        assert adj_matrix.shape[0] == adj_matrix.shape[1] == Nk

        # Setup save directory
        self.save_dir = Path(save_dir)
        if not self.save_dir.is_dir():
            os.mkdir(self.save_dir)

        # Setup logger
        if log_dir is not None:
            if not Path(log_dir).is_dir():
                os.mkdir(Path(log_dir))
            fname = datetime.datetime.now(tz=timezone.utc).isoformat()
            fname = fname.split('.')[0].replace(':', '.') + 'UTC_sys.log'
            f_handler = logging.FileHandler(Path(log_dir) / fname, mode='w', encoding='utf-8')
            f_handler.setLevel(logging.DEBUG)
            f_handler.setFormatter(FORMATTER)
            logger.addHandler(f_handler)
        self.logger = logger.getChild(self.__class__.__name__)

        # Store system info in a directed graph data structure
        self.graph = nx.DiGraph()
        self.exo_vars = exo_vars
        self.exo_bds = [rv.bounds() for rv in self.exo_vars]
        self.coupling_bds = coupling_bds
        self.adj_matrix = adj_matrix
        self.refine_level = 0

        # Construct graph nodes
        for k in range(Nk):
            # Count total number of inputs and get global indices of coupling inputs
            comp_dict = components[k]
            global_idx = []
            for comp_name, local_idx in comp_dict['local_in'].items():
                for k2 in range(Nk):
                    if components[k2]['name'] == comp_name:
                        global_idx.extend([components[k2]['global_out'][i] for i in local_idx])
            global_idx = sorted(global_idx)

            # Initialize a component surrogate (assume uniform dist for coupling vars)
            surr_type = comp_dict.get('type', 'lagrange')
            surr_class = None
            match surr_type:
                case 'lagrange':
                    surr_class = LagrangeSurrogate
                case other:
                    raise NotImplementedError(f"Surrogate type '{surr_type}' is not known at this time")
            x_vars = [self.exo_vars[i] for i in comp_dict['exo_in']] + [UniformRV(*coupling_bds[i]) for i in global_idx]
            surr = surr_class([], x_vars, comp_dict['model'], comp_dict['truth_alpha'],
                              max_alpha=comp_dict.get('max_alpha', None), max_beta=comp_dict.get('max_beta', 5))

            # Add the component as a str() node, with attributes specifying details of the surrogate
            self.graph.add_node(comp_dict['name'], exo_in=comp_dict['exo_in'], local_in=comp_dict['local_in'],
                                global_in=global_idx, global_out=comp_dict['global_out'], surrogate=surr,
                                is_computed=False)

        # Add directed edges according to the adjacency matrix
        for i in range(Nk):
            for j in range(Nk):
                if self.adj_matrix[i, j]:
                    self.graph.add_edge(components[i]['name'], components[j]['name'])

        # Estimate coupling variable bounds
        if est_bds > 0:
            self.estimate_coupling_bds(est_bds)

        self.init_surrogate()
        self.save_to_file('sys_init.pkl')

    def save_on_error(func):
        """Gracefully exit and save SystemSurrogate object on any errors"""
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except:
                self.save_to_file('sys_error.pkl')
                self.logger.critical(f'An error occurred during execution of {func.__name__}. Saving '
                                     f'SystemSurrogate object to sys_error.pkl', exc_info=True)
                self.logger.info(f'Final system surrogate on exit:\n {self}')
                raise
        return wrap
    save_on_error = staticmethod(save_on_error)

    @save_on_error
    def init_surrogate(self):
        """Add the coarsest multi-index to each component surrogate"""
        self.print_title_str('Initializing all component surrogates')
        for node, node_obj in self.graph.nodes.items():
            surr = node_obj['surrogate']
            alpha = (0,) * len(surr.truth_alpha)
            beta = (0,) * (len(node_obj['exo_in']) + len(node_obj['global_in']))
            surr.activate_index(alpha, beta)
            self.logger.info(f"Initialized component '{node}' with the multi-index {(alpha, beta)}. "
                             f"Runtime: {surr.get_cost(alpha, beta)} s")

    @save_on_error
    def build_surrogate(self, qoi_ind=None, N_refine=100, max_iter=20, max_tol=0.001, max_runtime=60, save_interval=0):
        """Build the system surrogate by iterative refinement until an end condition is met
        :param qoi_ind: list(), Indices of system QoI to focus refinement on, use all QoI if not specified
        :param N_refine: number of samples of exogenous inputs to compute error indicators on
        :param max_iter: the maximum number of refinement steps to take
        :param max_tol: the max allowable value in normalized RMSE to achieve (in units of 'pct of QoI mean')
        :param max_runtime: the maximum wall clock time (s) to run refinement for (will go until all models finish)
        :param save_interval (int) number of refinement steps between each progress save, none if 0
        """
        i = 0
        curr_error = np.inf
        t_start = time.time()
        while True:
            # Check all end conditions
            if i >= max_iter:
                self.print_title_str(f'Termination criteria reached: Max iteration {i}/{max_iter}')
                break
            if curr_error == -np.inf:
                self.print_title_str(f'Termination criteria reached: No candidates left to refine')
                break
            if curr_error < max_tol:
                self.print_title_str(f'Termination criteria reached: error {curr_error} < tol {max_tol}')
                break
            if time.time() - t_start >= max_runtime:
                actual = datetime.timedelta(seconds=time.time()-t_start)
                target = datetime.timedelta(seconds=max_runtime)
                self.print_title_str(f'Termination criteria reached: runtime {str(actual)} > {str(target)}')
                break

            curr_error = self.refine(qoi_ind=qoi_ind, N_refine=N_refine)
            if save_interval > 0 and self.refine_level % save_interval == 0:
                self.save_to_file(f'sys_iter_{self.refine_level}.pkl')
            i += 1

        self.save_to_file(f'sys_final.pkl')
        self.logger.info(f'Final system surrogate:\n {self}')

    def refine(self, qoi_ind=None, N_refine=100):
        """Find and refine the component surrogate with the largest error on system QoI
        :param qoi_ind: list(), Indices of system QoI to focus surrogate refinement on, use all QoI if not specified
        :param N_refine: number of samples of exogenous inputs to compute error indicators on
        """
        self.print_title_str(f'Refining system surrogate: iteration {self.refine_level+1}')
        if qoi_ind is None:
            qoi_ind = slice(None)

        # Construct a test set of random exogenous input samples
        xtest = self.sample_exo_inputs((N_refine,))

        # Initialize variables
        local_estimate = False
        y_curr = None
        y_min = None
        y_max = None
        y_curr_local = dict()
        x_couple = np.zeros((N_refine, len(self.coupling_bds)))

        # Sample coupling vars (uniform) and evaluate each component surrogate individually for local error estimation
        for i, bds in enumerate(self.coupling_bds):
            x_couple[:, i] = np.random.rand(N_refine) * (bds[1] - bds[0]) + bds[0]
        for node, node_obj in self.graph.nodes.items():
            comp_input = np.concatenate((xtest[:, node_obj['exo_in']], x_couple[:, node_obj['global_in']]), axis=1)
            y_curr_local[node] = node_obj['surrogate'](comp_input, training=True)

            # Determine if local error estimate is needed (if any component surrogate is still a constant approximation)
            if not local_estimate and np.any(np.isclose(np.std(y_curr_local[node], axis=0, keepdims=True), 0)):
                local_estimate = True

        # Compute entire integrated-surrogate for global system QoI error estimation
        if not local_estimate:
            y_curr = self(xtest, training=True)
            y_min = np.min(y_curr, axis=0, keepdims=True)  # (1, ydim)
            y_max = np.max(y_curr, axis=0, keepdims=True)  # (1, ydim)

        # Find the candidate surrogate with the largest error indicator
        error_max = -np.inf
        nrmse_max = -np.inf
        refine_indicator = -np.inf
        node_star = None
        alpha_star = None
        beta_star = None
        for node, node_obj in self.graph.nodes.items():
            self.logger.info(f"Estimating error for component '{node}'...")
            no_cand_flag = True
            for alpha, beta in node_obj['surrogate'].iterate_candidates():
                # Compute errors, ignoring NaN samples that may have resulted from bad FPI
                delta_error = None
                no_cand_flag = False
                if local_estimate:
                    # Use a local estimate of error to avoid initialization issues with system QoI
                    comp_input = np.concatenate((xtest[:, node_obj['exo_in']], x_couple[:, node_obj['global_in']]),
                                                axis=1)
                    y_cand = node_obj['surrogate'](comp_input, training=True)
                    y_curr = y_curr_local[node]
                    nrmse = np.sqrt(np.nanmean((y_cand - y_curr) ** 2, axis=0)) / np.abs(np.nanmean(y_curr, axis=0))
                    delta_error = np.nanmax(nrmse)  # Max normalized rmse over all current component outputs
                else:
                    # Otherwise, use a global system QoI estimate of error
                    y_cand = self(xtest, training=True)
                    y_min = np.min(np.concatenate((y_min, y_cand), axis=0), axis=0, keepdims=True)
                    y_max = np.max(np.concatenate((y_max, y_cand), axis=0), axis=0, keepdims=True)
                    error = y_cand[:, qoi_ind] - y_curr[:, qoi_ind]
                    nrmse = np.sqrt(np.nanmean(error ** 2, axis=0)) / np.abs(np.nanmean(y_curr[:, qoi_ind], axis=0))
                    delta_error = np.nanmax(nrmse)  # Max normalized rmse over all system QoIs

                # Ignore work in components that take less than 1 second to run (probably just analytical models)
                delta_work = max(1, node_obj['surrogate'].get_cost(alpha, beta))    # Wall time (s)
                refine_indicator = delta_error / delta_work
                self.logger.info(f"Candidate multi-index: {(alpha, beta)}. "
                                 f"{'(Local)' if local_estimate else '(Global)'} error indicator: {refine_indicator}")

                if refine_indicator > error_max:
                    error_max = refine_indicator
                    nrmse_max = delta_error
                    node_star = node
                    alpha_star = alpha
                    beta_star = beta

            if no_cand_flag:
                self.logger.info(f"Component '{node}' has no available candidates left!")

        # Update all coupling variable ranges
        if not local_estimate:
            for i in range(y_curr.shape[-1]):
                self.update_coupling_bds(i, (y_min[0, i], y_max[0, i]))

        # Add the chosen multi-index to the chosen component
        if node_star is not None:
            self.logger.info(f"Candidate multi-index {(alpha_star, beta_star)} chosen for component '{node_star}'")
            self.graph.nodes[node_star]['surrogate'].activate_index(alpha_star, beta_star)
            self.refine_level += 1
        else:
            self.logger.info(f"No candidates left for refinement, iteration: {self.refine_level}")

        return nrmse_max  # Probably a more intuitive metric for setting tolerances

    def __call__(self, x, max_fpi_iter=100, anderson_mem=10, fpi_tol=1e-10, ground_truth=False, verbose=False,
                 training=False):
        """Evaluate the system surrogate at exogenous inputs x
        :param x: (..., xdim) the points to be interpolated, must be within domain of self.exo_bds
        :param max_fpi_iter: the limit on convergence for the fixed-point iteration routine
        :param anderson_mem: hyperparameter for tuning the convergence of FPI with anderson acceleration
        :param fpi_tol: tolerance limit for convergence of fixed-point iteration
        :param ground_truth: whether to evaluate with the surrogates or the highest-fidelity 'ground truth' model
        :param verbose: whether to print out iteration progress during execution
        :param training: whether to call the system surrogate in training or evaluation mode
        :returns y: (..., ydim) the surrogate approximation of the system QoIs
        """
        # Allocate space for all system outputs (just save all coupling vars)
        xdim = x.shape[-1]
        ydim = len(self.coupling_bds)
        y = np.zeros(x.shape[:-1] + (ydim,))
        valid_idx = ~np.isnan(x[..., 0])  # Keep track of valid samples (set to False if FPI fails)
        t1 = 0

        # Initialize all components
        for node, node_obj in self.graph.nodes.items():
            node_obj['is_computed'] = False

        # Convert system into DAG by grouping strongly-connected-components
        dag = nx.condensation(self.graph)

        # Compute component models in topological order
        for supernode in nx.topological_sort(dag):
            scc = [n for n in dag.nodes[supernode]['members']]

            # Compute single component feedforward output (no FPI needed)
            if len(scc) == 1:
                if verbose:
                    self.logger.info(f"Running component '{scc[0]}'...")
                    t1 = time.time()

                # Gather inputs
                node_obj = self.graph.nodes[scc[0]]
                exo_inputs = x[..., node_obj['exo_in']]
                for comp_name in node_obj['local_in']:
                    assert self.graph.nodes[comp_name]['is_computed']
                coupling_inputs = y[..., node_obj['global_in']]
                comp_input = np.concatenate((exo_inputs, coupling_inputs), axis=-1)  # (..., xdim)

                # Compute outputs
                comp_output = node_obj['surrogate'](comp_input[valid_idx, :], ground_truth=ground_truth,
                                                    training=training)
                for local_i, global_i in enumerate(node_obj['global_out']):
                    y[valid_idx, global_i] = comp_output[..., local_i]
                node_obj['is_computed'] = True

                if verbose:
                    self.logger.info(f"Component '{scc[0]}' completed. Runtime: {time.time() - t1} s")

            # Handle FPI for SCCs with more than one component
            else:
                # Set the initial guess for all coupling vars (middle of domain)
                x_couple = np.array([(bds[0] + bds[1]) / 2 for bds in self.coupling_bds])
                x_couple = np.broadcast_to(x_couple, x.shape[:-1] + x_couple.shape).copy()

                adj_nodes = []
                fpi_idx = set()
                for node in scc:
                    for comp_name, local_idx in self.graph.nodes[node]['local_in'].items():
                        # Track the global idx of all coupling vars that need FPI
                        if comp_name in scc:
                            fpi_idx.update([self.graph.nodes[comp_name]['global_out'][idx] for idx in local_idx])

                        # Override coupling vars from components outside the scc (should already be computed)
                        if comp_name not in scc and comp_name not in adj_nodes:
                            assert self.graph.nodes[comp_name]['is_computed']
                            global_idx = self.graph.nodes[comp_name]['global_out']
                            x_couple[..., global_idx] = y[..., global_idx]
                            adj_nodes.append(comp_name)  # Only need to do this once for each adj component
                x_couple_next = x_couple.copy()
                fpi_idx = sorted(fpi_idx)

                # Main FPI loop
                if verbose:
                    self.logger.info(f"Initializing FPI for SCC {scc} ...")
                    t1 = time.time()
                k = 0
                residual_hist = None
                x_hist = None
                while True:
                    for node in scc:
                        # Gather inputs from exogenous and coupling sources
                        node_obj = self.graph.nodes[node]
                        exo_inputs = x[..., node_obj['exo_in']]
                        coupling_inputs = x_couple[..., node_obj['global_in']]
                        comp_input = np.concatenate((exo_inputs, coupling_inputs), axis=-1)     # (..., xdim)

                        # Compute component outputs
                        comp_output = node_obj['surrogate'](comp_input[valid_idx, :], ground_truth=ground_truth,
                                                            training=training)
                        global_idx = node_obj['global_out']
                        for local_i, global_i in enumerate(global_idx):
                            x_couple_next[valid_idx, global_i] = comp_output[..., local_i]
                            # Can't splice valid_idx with global_idx for some reason, have to loop over global_idx here

                    # Compute residual and check end conditions
                    residual = np.expand_dims(x_couple_next[..., fpi_idx] - x_couple[..., fpi_idx], axis=-1)
                    max_error = np.max(np.abs(residual[valid_idx, :, :]))
                    if verbose:
                        self.logger.info(f'FPI iter: {k}. Max residual: {max_error}. Time: {time.time() - t1} s')
                    if max_error <= fpi_tol:
                        if verbose:
                            self.logger.info(f'FPI converged for SCC {scc} in {k} iterations with {max_error} < tol '
                                             f'{fpi_tol}. Final time: {time.time() - t1} s')
                        break
                    if k >= max_fpi_iter:
                        self.logger.warning(f'FPI did not converge in {max_fpi_iter} iterations for SCC {scc}: '
                                            f'{max_error} > tol {fpi_tol}. Some samples will be returned as NaN.')
                        converged_idx = np.max(np.abs(residual), axis=(-1, -2)) <= fpi_tol
                        for idx in fpi_idx:
                            y[~converged_idx, idx] = np.nan
                        valid_idx = np.logical_and(valid_idx, converged_idx)
                        break

                    # Keep track of residual and x_couple histories
                    if k == 0:
                        residual_hist = residual.copy()                                 # (..., xdim, 1)
                        x_hist = np.expand_dims(x_couple_next[..., fpi_idx], axis=-1)   # (..., xdim, 1)
                        x_couple[:] = x_couple_next[:]
                        k += 1
                        continue  # skip anderson accel on first iteration

                    # Iterate with anderson acceleration (only iterate on samples that are not yet converged)
                    converged_idx = np.max(np.abs(residual), axis=(-1, -2)) <= fpi_tol
                    curr_idx = np.logical_and(valid_idx, ~converged_idx)
                    residual_hist = np.concatenate((residual_hist, residual), axis=-1)
                    x_hist = np.concatenate((x_hist, np.expand_dims(x_couple_next[..., fpi_idx], axis=-1)), axis=-1)
                    mk = min(anderson_mem, k)
                    Fk = residual_hist[curr_idx, :, k-mk:]                               # (..., xdim, mk+1)
                    C = np.ones(Fk.shape[:-2] + (1, mk + 1))
                    b = np.zeros(Fk.shape[:-2] + (len(fpi_idx), 1))
                    d = np.ones(Fk.shape[:-2] + (1, 1))
                    alpha = np.expand_dims(self.constrained_lls(Fk, b, C, d), axis=-3)   # (..., 1, mk+1, 1)
                    x_new = np.squeeze(x_hist[curr_idx, :, np.newaxis, -(mk+1):] @ alpha, axis=(-1, -2))
                    for local_i, global_i in enumerate(fpi_idx):
                        x_couple[curr_idx, global_i] = x_new[..., local_i]
                    k += 1

                # Save outputs of each component in SCC after convergence of FPI
                for node in scc:
                    global_idx = self.graph.nodes[node]['global_out']
                    for global_i in global_idx:
                        y[valid_idx, global_i] = x_couple_next[valid_idx, global_i]
                    self.graph.nodes[node]['is_computed'] = True

        # Return all component outputs (..., ydim), samples that didn't converge during FPI are left as np.nan
        return y

    def estimate_coupling_bds(self, N_est, anderson_mem=10, fpi_tol=1e-10, max_fpi_iter=100):
        """Estimate and set the coupling variable bounds
        :param N_est: The number of samples of exogenous inputs to use
        """
        self.print_title_str('Estimating coupling variable bounds')
        x = self.sample_exo_inputs((N_est,))
        y = self(x, ground_truth=True, verbose=True, anderson_mem=anderson_mem, fpi_tol=fpi_tol,
                 max_fpi_iter=max_fpi_iter)
        for i in range(len(self.coupling_bds)):
            lb = np.min(y[:, i])
            ub = np.max(y[:, i])
            self.update_coupling_bds(i, (lb, ub), init=True)

    def update_coupling_bds(self, global_idx, bds, init=False, buffer=0.05):
        """Update coupling variable bounds
        :param global_idx: index of coupling variable to update
        :param bds: tuple() specifying bounds to update the current bounds with
        :param init: whether to set new bounds or update existing (default)
        :param buffer: Fraction of domain length to buffer upper/lower bounds
        """
        offset = buffer * (bds[1] - bds[0])
        offset_bds = (bds[0] - offset, bds[1] + offset)
        new_bds = offset_bds if init else (min(self.coupling_bds[global_idx][0], offset_bds[0]),
                                           max(self.coupling_bds[global_idx][1], offset_bds[1]))
        self.coupling_bds[global_idx] = new_bds

        # Iterate over all components and update internal coupling variable bounds
        for node_name, node_obj in self.graph.nodes.items():
            if global_idx in node_obj['global_in']:
                # Get the local index for this coupling variable within each component's inputs
                local_idx = len(node_obj['exo_in']) + node_obj['global_in'].index(global_idx)
                node_obj['surrogate'].update_input_bds(local_idx, new_bds)

    def sample_exo_inputs(self, shape):
        """Return samples of the exogenous inputs
        :param shape: tuple() specifying shape of the samples
        :returns x: (*shape, xdim) samples of the exogenous inputs
        """
        xdim = len(self.exo_vars)
        x = np.zeros((*shape, xdim))
        for i in range(xdim):
            x[..., i] = self.exo_vars[i].sample(shape)

        return x

    def get_component(self, comp_name):
        """Return the ComponentSurrogate object for this component
        :param comp_name: (str) name of the component to return
        """
        return self.graph.nodes[comp_name]['surrogate']

    def print_title_str(self, title_str):
        """Quick wrapper to log an important message"""
        self.logger.info('-' * int(len(title_str)/2) + title_str + '-' * int(len(title_str)/2))

    def save_to_file(self, filename):
        with open(self.save_dir / filename, 'wb') as dill_file:
            dill.dump(self, dill_file)

    def __getitem__(self, component):
        """Convenience method to get the ComponentSurrogate object from the system
        :param component: (str) name of the component to get
        """
        return self.graph.nodes[component]['surrogate']

    def __repr__(self):
        s = f'----SystemSurrogate----\nAdjacency: \n{self.adj_matrix}\nExogenous inputs: {self.exo_vars}\n'
        for node, node_obj in self.graph.nodes.items():
            s += f'Component: {node}\n{node_obj["surrogate"]}'
        return s

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'rb') as dill_file:
            sys = dill.load(dill_file)

        return sys

    @staticmethod
    def constrained_lls(A, b, C, d):
        """Minimize ||Ax-b||_2, subject to Cx=d
        :param A: (..., M, N)
        :param b: (..., M, 1)
        :param C: (..., P, N)
        :param d: (..., P, 1)
        """
        M = A.shape[-2]
        dims = len(A.shape[:-2])
        T_axes = tuple(np.arange(0, dims)) + (-1, -2)
        Q, R = np.linalg.qr(np.concatenate((A, C), axis=-2))
        Q1 = Q[..., :M, :]
        Q2 = Q[..., M:, :]
        Q1_T = np.transpose(Q1, axes=T_axes)
        Q2_T = np.transpose(Q2, axes=T_axes)
        Qtilde, Rtilde = np.linalg.qr(Q2_T)
        Qtilde_T = np.transpose(Qtilde, axes=T_axes)
        Rtilde_T_inv = np.linalg.pinv(np.transpose(Rtilde, axes=T_axes))
        w = np.linalg.pinv(Rtilde) @ (Qtilde_T @ Q1_T @ b - Rtilde_T_inv @ d)

        return np.linalg.pinv(R) @ (Q1_T @ b - Q2_T @ w)


class ComponentSurrogate(ABC):
    """Multi-index stochastic collocation (MISC) surrogate abstract class for a component model"""

    def __init__(self, multi_index, x_vars, model, truth_alpha, max_alpha=None, max_beta=5):
        """Construct the MISC surrogate from a multi-index
        :param multi_index: [((alpha1), (beta1)), ... ] List of concatenated multi-indices alpha, beta specifying
                            model and surrogate fidelity
        :param x_vars: [X1, X2, ...] list of BaseRV() objects specifying bounds/pdfs for each input x
        :param model: The function to approximate, callable as y = model(x, alpha)
        :param truth_alpha: tuple() specifying the highest model fidelity indices necessary for a 'truth' comparison
        :param max_alpha: tuple(), the maximum model refinement indices to allow
        :param max_beta: (int), the maximum surrogate refinement level in any one input dimension
        """
        self.logger = logger.getChild(self.__class__.__name__)
        assert self.is_downward_closed(multi_index), 'Must be a downward closed set.'
        self.index_set = []         # The active index set for the MISC approximation
        self.candidate_set = []     # Candidate indices for refinement
        self._model = model
        self.truth_alpha = truth_alpha
        self._truth_model = lambda x: self._model(x, self.truth_alpha)
        self.x_vars = x_vars
        self.x_bds = [rv.bounds() for rv in self.x_vars]
        self.ydim = None
        self.ij = None
        self.index_mat = None
        max_alpha = (3,)*len(truth_alpha) if max_alpha is None else max_alpha
        self.max_refine = list(max_alpha + (max_beta,)*len(self.x_vars))    # Max refinement indices

        # Construct vectors of [0,1]^dim(alpha+beta), used for combination coefficients
        Nij = len(truth_alpha) + len(self.x_vars)
        self.ij = np.zeros((2 ** Nij, Nij), dtype=int)
        for i, ele in enumerate(itertools.product([0, 1], repeat=Nij)):
            self.ij[i, :] = ele

        # This is a (N_indices, dim(alpha+beta)) refactor of self.index_set, useful for arrayed computations
        self.index_mat = np.zeros((0, Nij), dtype=int)

        # Initialize important tree-like structures
        self.surrogates = dict()        # Maps alphas -> betas -> surrogates
        self.costs = dict()             # Maps alphas -> betas -> wall clock run times
        self.alpha_models = dict()      # Maps alphas -> models

        # Initialize any indices that were passed in
        for alpha, beta in multi_index:
            self.activate_index(alpha, beta)

    def activate_index(self, alpha, beta):
        """Add a multi-index to the active set and all neighbors to the candidate set
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        """
        # User is responsible for making sure index set is downward-closed
        self.add_surrogate(alpha, beta)
        ele = (alpha, beta)

        # Add all possible new candidates (distance of one unit vector away)
        ind = list(alpha + beta)
        new_candidates = []
        for i in range(len(ind)):
            ind_new = ind.copy()
            ind_new[i] += 1

            # Don't add if we surpass a refinement limit
            if np.any(np.array(ind_new) > np.array(self.max_refine)):
                continue

            # Add the new index if it maintains downward-closedness
            down_closed = True
            for j in range(len(ind)):
                ind_check = ind_new.copy()
                ind_check[j] -= 1
                if ind_check[j] >= 0:
                    tup_check = (tuple(ind_check[:len(alpha)]), tuple(ind_check[len(alpha):]))
                    if tup_check not in self.index_set and tup_check != ele:
                        down_closed = False
                        break
            if down_closed:
                new_cand = (tuple(ind_new[:len(alpha)]), tuple(ind_new[len(alpha):]))
                self.add_surrogate(*new_cand)
                new_candidates.append(new_cand)

        # Move to the active index set
        if ele in self.candidate_set:
            self.candidate_set.remove(ele)
        self.index_set.append(ele)
        self.index_mat = np.concatenate((self.index_mat, np.array(alpha + beta)[np.newaxis, :]), axis=0)
        new_candidates = [cand for cand in new_candidates if cand not in self.candidate_set]
        self.candidate_set.extend(new_candidates)

    def add_surrogate(self, alpha, beta):
        """Build a BaseInterpolator object for a given alpha, beta index
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        """
        # Store a new function for each unique fidelity (alpha) of the main forward model
        if str(alpha) not in self.alpha_models:
            self.alpha_models[str(alpha)] = lambda x: self._model(x, alpha)

        # Create a dictionary for each alpha model to store multiple surrogate fidelities (beta)
        if str(alpha) not in self.surrogates:
            self.surrogates[str(alpha)] = dict()
            self.costs[str(alpha)] = dict()

        # Create a new interpolator object for this multi-index (abstract method)
        if self.surrogates[str(alpha)].get(str(beta), None) is None:
            self.logger.info(f'Building interpolant for index {(alpha, beta)} ...')
            interp, cost = self.add_interpolator(alpha, beta)
            self.surrogates[str(alpha)][str(beta)] = interp
            self.costs[str(alpha)][str(beta)] = cost
            self.ydim = interp.get_ydim()

    def iterate_candidates(self):
        """Iterate candidate indices one by one into the active index set
        :yields alpha, beta: the multi-indices of the current candidate that has been moved to active set
        """
        for alpha, beta in self.candidate_set:
            # Temporarily add a candidate index to active set
            self.index_set.append((alpha, beta))
            self.index_mat = np.concatenate((self.index_mat, np.array(alpha+beta)[np.newaxis, :]), axis=0)
            yield alpha, beta
            self.index_set.remove((alpha, beta))
            self.index_mat = self.index_mat[:-1, :]

    def __call__(self, x, ground_truth=False, training=False):
        """Evaluate the surrogate at points x
        :param x: (..., xdim) the points to be interpolated, must be within domain of self.x_bds
        :param ground_truth: whether to use the highest fidelity model or the surrogate (default)
        :param training: if True, then only compute with active index set, otherwise use all candidates as well
        :returns y: (..., ydim) the surrogate approximation of the qois
        """
        if ground_truth:
            # Bypass surrogate evaluation
            return self._truth_model(x)

        index_set = self.index_set if training else self.index_set + self.candidate_set
        # assert self.is_downward_closed(index_set)

        if not training:
            # Add candidate indices to MISC approximation if not training
            cand_indices = np.zeros((len(self.candidate_set), self.index_mat.shape[1]))
            for i, (alpha, beta) in enumerate(self.candidate_set):
                cand_indices[i, :] = alpha + beta
            self.index_mat = np.concatenate((self.index_mat, cand_indices), axis=0)

        y = np.zeros(x.shape[:-1] + (self.ydim,))
        for alpha, beta in index_set:
            comb_coeff = self.compute_misc_coefficient(alpha, beta)
            if np.abs(comb_coeff) > 0:
                func = self.surrogates[str(alpha)][str(beta)]
                y += comb_coeff * func(x)

        if not training:
            sl = slice(0, -len(self.candidate_set)) if len(self.candidate_set) else slice(None)
            self.index_mat = self.index_mat[sl, :]

        return y

    def compute_misc_coefficient(self, alpha, beta):
        """Compute combination technique formula for MISC
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        """
        # Add permutations of [0, 1] to (alpha, beta)
        alpha_beta = np.array(alpha+beta, dtype=int)[np.newaxis, :]     # (1, Nij)
        new_indices = np.expand_dims(alpha_beta + self.ij, axis=1)      # (2**Nij, 1, Nij)
        index_set = np.expand_dims(self.index_mat, axis=0)              # (1, Ns, Nij)

        # Find which indices are in self.index_set (using np broadcasting comparison)
        diff = new_indices - index_set                                  # (2**Nij, Ns, Nij)
        idx = np.count_nonzero(diff, axis=-1) == 0                      # (2**Nij, Ns)
        idx = np.any(idx, axis=-1)                                      # (2**Nij,)
        ij_use = self.ij[idx, :]                                        # (*, Nij)
        l1_norm = np.sum(np.abs(ij_use), axis=-1)                       # (*,)

        return np.sum((-1) ** l1_norm)

    def get_sub_surrogate(self, alpha, beta):
        """Get the specific sub-surrogate corresponding to the (alpha,beta) fidelity
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        """
        return self.surrogates[str(alpha)][str(beta)]

    def get_cost(self, alpha, beta):
        """Return the total cost (wall time s) required to add (alpha, beta) to the MISC approximation
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        """
        return self.costs[str(alpha)][str(beta)]

    def update_input_bds(self, idx, bds):
        """Update the bounds of the input at the given idx (assumes a uniform RV)
        :param idx: the index of the input variable to update
        :param bds: tuple() specifying the new bounds to update
        """
        self.x_vars[idx].update_bounds(*bds)
        self.x_bds[idx] = bds

        # Update the bounds in all associated tensor-product surrogates
        for alpha in self.surrogates:
            for beta in self.surrogates[alpha]:
                self.surrogates[alpha][beta].update_input_bds(idx, bds)

    def __repr__(self):
        s = f'Inputs \u2014 {self.x_vars}\n'
        for alpha, beta in self.index_set:
            s += f"[{int(self.compute_misc_coefficient(alpha, beta))}] \u2014 {alpha}, {beta}\n"
        for alpha, beta in self.candidate_set:
            s += f"[-] \u2014 {alpha}, {beta}\n"
        return s

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def is_one_level_refinement(beta_old, beta_new):
        """Check if a new beta multi-index (tuple) is a one-level refinement from a previous beta"""
        level_diff = np.array(beta_new, dtype=int) - np.array(beta_old, dtype=int)
        ind = np.nonzero(level_diff)[0]
        return ind.shape[0] == 1 and level_diff[ind] == 1

    @staticmethod
    def is_downward_closed(indices):
        """Return if a list of (alpha, beta) multi-indices is downward closed
        :param indices: list() of (alpha, beta) multi-indices
        """
        # Iterate over every multi-index
        for alpha, beta in indices:
            # Every smaller multi-index must also be included in the indices list
            sub_sets = [np.arange(tuple(alpha+beta)[i]+1) for i in range(len(alpha) + len(beta))]
            for ele in itertools.product(*sub_sets):
                tup = (tuple(ele[:len(alpha)]), tuple(ele[len(alpha):]))
                if tup not in indices:
                    return False
        return True

    @abstractmethod
    def add_interpolator(self, alpha, beta):
        """Return a BaseInterpolator object and the wall time cost for a given alpha, beta index
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        :returns interp, cost: the BaseInterpolator object and the wall time (s) required to construct it
        """
        pass


class BaseInterpolator(ABC):
    """Base interpolator abstract class"""

    def __init__(self, beta, x_vars, xi=None, yi=None, model=None):
        """Construct the interpolator
        :param beta: list(), refinement level in each dimension of xdim
        :param x_vars: list() of BaseRV() objects specifying bounds/pdfs for each input x
        :param xi: (Nx, xdim) interpolation points
        :param yi: the interpolation qoi values, y = (Nx, ydim)
        :param model: Callable as y = model(x), with x = (..., xdim), y = (..., ydim)
        """
        self.logger = logger.getChild(self.__class__.__name__)
        self._model = model
        self.xi = xi                                        # Interpolation points
        self.yi = yi                                        # Function values at interpolation points
        self.beta = beta                                    # Refinement level indices
        self.x_vars = x_vars                                # BaseRV() objects for each input
        self.x_bds = [rv.bounds() for rv in self.x_vars]    # Bounds on inputs
        self.wall_time = 1                                  # Wall time to evaluate model (s)
        self.xdim = len(self.x_vars)
        assert len(beta) == self.xdim

    def update_input_bds(self, idx, bds):
        """Update the input bounds at the given index (assume a uniform RV)"""
        self.x_vars[idx].update_bounds(*bds)
        self.x_bds[idx] = bds

    def get_ydim(self):
        """Get the dimension of the outputs"""
        return self.yi.shape[-1] if self.yi is not None else None

    def set_yi(self, yi=None, model=None):
        """Set the interpolation point qois, if yi is none then compute yi=model(self.xi)
        :param yi: (Nx, ydim) must match dimension of self.xi
        :param model: Callable as y = model(x), with x = (..., xdim), y = (..., ydim)
        """
        if model is not None:
            self._model = model
        if yi is None:
            if self._model is None:
                error_msg = 'Model not specified for computing QoIs at interpolation grid points.'
                self.logger.error(error_msg)
                raise Exception(error_msg)
            t1 = time.time()
            self.yi = self._model(self.xi)
            self.wall_time = (time.time() - t1) / self.xi.shape[0]
        else:
            self.yi = yi

    @abstractmethod
    def refine(self, beta, manual=False):
        """Return a new interpolator with one dimension refined by one level, specified by beta
        :param beta: list(), The new refinement level, should only refine one dimension
        :param manual: whether to manually compute model at refinement points
        :return interp: a refined BaseInterpolator object
             or x_new_idx, x_new, interp: where x_new are the newly refined interpolation points (N_new, xdim) and
                                          x_new_idx is the list of indices of these points into interp.xi and interp.yi,
                                          Would use this if you did not provide a callable model to the Interpolator or
                                          you want to manually set yi for each new xi outside this function
        """
        pass

    @abstractmethod
    def __call__(self, x):
        """Evaluate the interpolation at points x
        :param x: (..., xdim) the points to be interpolated, must be within domain of self.xi
        :returns y: (..., ydim) the interpolated value of the qois
        """
        pass
