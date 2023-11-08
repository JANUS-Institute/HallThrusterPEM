"""Module for system-level adaptive multidisciplinary, multi-fidelity surrogate implementation"""
import logging
import numpy as np
import networkx as nx
import itertools
import sys
import os
import time
import datetime
import functools
import dill
import copy
from abc import ABC, abstractmethod
from datetime import timezone
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from joblib.externals.loky import set_loky_pickler

sys.path.append('..')

from utils import get_logger, ax_default


class SystemSurrogate:

    def __init__(self, components, exo_vars, coupling_vars, est_bds=0, root_dir=None, executor=None,
                 stdout=True, init_surr=True):
        """Construct a multidisciplinary system surrogate
        :param components: list(Nk,) of dicts specifying component name (str), a callable function model(x, alpha), a
                           string specifying surrogate class type, the highest 'truth' set of model fidelity indices,
                           the maximum level of model fidelity indices, the maximum surrogate refinement level, the
                           global indices of required system exogenous inputs, a dict or list specifying the indices of
                           coupling variable inputs from other models, a list() that maps local component outputs
                           to global coupling indices, whether to save all model outputs to file, and model args/kwargs
                           Ex: [{'name': 'Thruster', 'model': callable(x, alpha), 'truth_alpha': (3,), 'max_alpha': (3),
                                         'max_beta': (5,), 'exo_in': [0, 2, 6...], 'coupling_in': {'model1': [1,2,3],
                                         'model2': [0]}, 'coupling_out': [0, 1, 2, 3], 'type': 'lagrange',
                                         'save_output': True, 'model_args': (), 'model_kwargs': {}},
                                {'name': 'Plume', ...}, ...]
        :param exo_vars: list() of BaseRVs specifying the bounds/distributions for all system-level exogenous inputs
        :param coupling_vars: list() of UniformRVs specifying estimated bounds for all coupling variables
        :param est_bds: (int) number of samples to estimate coupling variable bounds, do nothing if 0
        :param root_dir: (str) Root directory for all build products (.logs, .pkl, .json, etc.)
        :param executor: An instance of a concurrent.futures.Executor, use to iterate new candidates in parallel
        :param stdout: only log to file and don't print to console if False
        :param init_surr: whether to initialize the surrogate with the coarsest fidelity index
        """
        # Setup root directory
        if root_dir is None:
            timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
            self.root_dir = 'build_' + timestamp
            os.mkdir(self.root_dir)
            self.root_dir = str(Path(self.root_dir).resolve())
        else:
            self.root_dir = str(Path(root_dir).resolve())
            if len(os.listdir(self.root_dir)) > 0:
                user_input = input(f'The contents of directory "{self.root_dir}" will be cleaned. Continue? (Y/n): ')
                if user_input.lower().startswith('y') or user_input == '' or user_input == '\n':
                    for root, dirs, files in os.walk(self.root_dir):
                        for f in files:
                            os.unlink(os.path.join(self.root_dir, f))
                        for d in dirs:
                            shutil.rmtree(os.path.join(self.root_dir, d))
                else:
                    error_msg = f'Please specify a different root directory to use then...'
                    print(error_msg)
                    raise Exception(error_msg)

        os.mkdir(Path(self.root_dir) / 'sys')
        os.mkdir(Path(self.root_dir) / 'components')
        fname = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.') + 'UTC_sys.log'
        self.log_file = str((Path(self.root_dir) / fname).resolve())
        self.logger = get_logger(self.__class__.__name__, log_file=self.log_file, stdout=stdout)
        self.executor = executor

        # Store system info in a directed graph data structure
        self.graph = nx.DiGraph()
        self.exo_vars = copy.deepcopy(exo_vars)
        self.coupling_vars = copy.deepcopy(coupling_vars)
        self.refine_level = 0
        self.build_metrics = dict()     # Save refinement error metrics during build

        # Construct graph nodes
        Nk = len(components)
        nodes = {comp['name']: comp for comp in components}  # work-around since self.graph.nodes is not built yet
        for k in range(Nk):
            # Add the component as a str() node, with attributes specifying details of the surrogate
            comp_dict = components[k]
            indices, surr = self.build_component(comp_dict, nodes=nodes)
            self.graph.add_node(comp_dict['name'], exo_in=indices['exo_in'], local_in=indices['local_in'],
                                global_in=indices['global_in'], global_out=indices['global_out'], surrogate=surr,
                                is_computed=False)

        # Connect all neighbor nodes
        for node, node_obj in self.graph.nodes.items():
            for neighbor in node_obj['local_in']:
                self.graph.add_edge(neighbor, node)

        # Estimate coupling variable bounds
        if est_bds > 0:
            self.estimate_coupling_bds(est_bds)

        # Init system with most coarse fidelity indices in each component
        if init_surr:
            self.init_system()
        self.save_to_file('sys_init.pkl')

    def build_component(self, component, nodes=None):
        """Build and return a ComponentSurrogate object from a dict that describes the component model/connections
        :param component: dict() specifying details of a component (see docs of __init__)
        :param nodes: dict() of {node: node_attributes}, defaults to self.graph.nodes
        :returns connections, surr: a dict() of all connection indices and the ComponentSurrogate object
        """
        nodes = self.graph.nodes if nodes is None else nodes
        args = component.get('model_args', ())
        kwargs = component.get('model_kwargs', {})

        # Get exogenous input indices (might already be a list of ints, otherwise convert list of vars to indices)
        exo_in = component.get('exo_in') if component.get('exo_in') and isinstance(component['exo_in'][0], int) else \
            [self.exo_vars.index(var) for var in component.get('exo_in')]

        # Get global coupling output indices for all nodes (convert list of vars to list of indices if necessary)
        global_out = {}
        for node, node_obj in nodes.items():
            global_out[node] = node_obj.get('coupling_out') if (node_obj.get('coupling_out') and
                                                                isinstance(node_obj['coupling_out'][0], int)) else \
                [self.coupling_vars.index(var) for var in node_obj.get('coupling_out')]

        # Refactor coupling inputs into both local and global index formats
        local_in = dict()   # e.g. {'Cathode': [0, 1, 2], 'Thruster': [0,], etc...}
        global_in = list()  # e.g. [0, 2, 4, 5, 6]
        if isinstance(component.get('coupling_in'), dict):
            # If already a dict, get local connection indices from each neighbor
            for node, connections in component['coupling_in'].items():
                if isinstance(connections[0], int):
                    local_in[node] = sorted(connections)
                else:
                    global_ind = [self.coupling_vars.index(var) for var in connections]
                    local_in[node] = sorted([global_out[node].index(i) for i in global_ind])

            # Convert to global coupling indices
            for node, local_idx in local_in.items():
                global_in.extend([global_out[node][i] for i in local_idx])
            global_in = sorted(global_in)
        else:
            # Otherwise, convert a list of global indices or vars into a dict of local indices
            global_in = sorted(component.get('coupling_in')) if (component.get('coupling_in') and
                                                                 isinstance(component['coupling_in'][0], int)) else \
                sorted([self.coupling_vars.index(var) for var in component.get('coupling_in')])
            for node, node_obj in nodes.items():
                l = list()
                for i in global_in:
                    try:
                        l.append(global_out[node].index(i))
                    except ValueError:
                        pass
                if l:
                    local_in[node] = sorted(l)

        # Store all connection indices for this component
        connections = dict(exo_in=exo_in, local_in=local_in, global_in=global_in,
                           global_out=global_out.get(component.get('name')))

        # Set up a component output save directory
        if component.get('save_output', False):
            output_dir = str((Path(self.root_dir) / 'components' / component['name']).resolve())
            if not Path(output_dir).is_dir():
                os.mkdir(output_dir)
            kwargs['output_dir'] = output_dir

        # Initialize a new component surrogate
        surr_type = component.get('type', 'lagrange')
        surr_class = None
        match surr_type:
            case 'lagrange':
                from surrogates.sparse_grids import SparseGridSurrogate
                surr_class = SparseGridSurrogate
            case 'analytical':
                surr_class = AnalyticalSurrogate
            case other:
                raise NotImplementedError(f"Surrogate type '{surr_type}' is not known at this time.")

        # Check for an override of model fidelity indices (to enable just single-fidelity evaluation)
        if kwargs.get('hf_override', False):
            truth_alpha, max_alpha = (), ()
            kwargs['hf_override'] = component['truth_alpha']    # Pass in the truth alpha indices as a kwarg to model
        else:
            truth_alpha, max_alpha = component['truth_alpha'], component.get('max_alpha', None)

        # Assumes input ordering is exogenous vars + sorted coupling vars
        x_vars = [self.exo_vars[i] for i in exo_in] + [self.coupling_vars[i] for i in global_in]
        surr = surr_class([], x_vars, component['model'], truth_alpha, max_alpha=max_alpha,
                          max_beta=component.get('max_beta', None), executor=self.executor, log_file=self.log_file,
                          model_args=args, model_kwargs=kwargs)
        return connections, surr

    def swap_component(self, component, exo_add=None, exo_remove=None, qoi_add=None, qoi_remove=None):
        """Swap a new component into the system, updating all connections/inputs
        :param component: dict() specifying component model and connections (see docs of __init__ for details)
        :param exo_add: list() of BaseRVs to add to system exogenous inputs (will be appended to end)
        :param exo_remove: list() of indices of system exogenous inputs to delete (can't be shared by other components)
        :param qoi_add: list() of UniformRVs specifying bounds of system output QoIs to add
        :param qoi_remove: list() of indices of system qois to delete (can't be shared by other components)
        """
        # Delete system exogenous inputs
        if exo_remove is None:
            exo_remove = []
        exo_remove = exo_remove if exo_remove and isinstance(exo_remove[0], int) else \
            [self.exo_vars.index(str(var)) for var in exo_remove]
        exo_remove = sorted(exo_remove)
        for j, exo_var_idx in enumerate(exo_remove):
            # Adjust exogenous indices for all components to account for deleted system inputs
            for node, node_obj in self.graph.nodes.items():
                if node != component['name']:
                    for i, idx in enumerate(node_obj['exo_in']):
                        if idx == exo_var_idx:
                            error_msg = f"Can't delete system exogenous input at idx {exo_var_idx}, since it is " \
                                        f"shared by component '{node}'."
                            self.logger.error(error_msg)
                            raise Exception(error_msg)
                        if idx > exo_var_idx:
                            node_obj['exo_in'][i] -= 1

            # Need to update the remaining delete indices by -1 to account for each sequential deletion
            del self.exo_vars[exo_var_idx]
            for i in range(j+1, len(exo_remove)):
                exo_remove[i] -= 1

        # Append any new exogenous inputs to the end
        if exo_add is not None:
            self.exo_vars.extend(exo_add)

        # Delete system qoi outputs (if not shared by other components)
        if qoi_remove is None:
            qoi_remove = []
        qoi_remove = qoi_remove if qoi_remove and isinstance(qoi_remove[0], int) else \
            [self.coupling_vars.index(str(var)) for var in qoi_remove]
        qoi_remove = sorted(qoi_remove)
        for j, qoi_idx in enumerate(qoi_remove):
            # Adjust coupling indices for all components to account for deleted system outputs
            for node, node_obj in self.graph.nodes.items():
                if node != component['name']:
                    for i, idx in enumerate(node_obj['global_in']):
                        if idx == qoi_idx:
                            error_msg = f"Can't delete system QoI at idx {qoi_idx}, since it is an input to " \
                                        f"component '{node}'."
                            self.logger.error(error_msg)
                            raise Exception(error_msg)
                        if idx > qoi_idx:
                            node_obj['global_in'][i] -= 1

                    for i, idx in enumerate(node_obj['global_out']):
                        if idx > qoi_idx:
                            node_obj['global_out'][i] -= 1

            # Need to update the remaining delete indices by -1 to account for each sequential deletion
            del self.coupling_vars[qoi_idx]
            for i in range(j+1, len(qoi_remove)):
                qoi_remove[i] -= 1

        # Append any new system QoI outputs to the end
        if qoi_add is not None:
            self.coupling_vars.extend(qoi_add)

        # Make changes to adj matrix if coupling inputs changed
        prev_neighbors = list(self.graph.nodes[component['name']]['local_in'].keys())
        new_neighbors = list(component['local_in'].keys())
        for neighbor in new_neighbors:
            if neighbor not in prev_neighbors:
                self.graph.add_edge(neighbor, component['name'])
            else:
                prev_neighbors.remove(neighbor)
        for neighbor in prev_neighbors:
            self.graph.remove_edge(neighbor, component['name'])

        # Build and initialize the new component surrogate
        indices, surr = self.build_component(component)
        surr.init_coarse()
        self.logger.info(f"Swapped component '{component['name']}'.")
        nx.set_node_attributes(self.graph, {component['name']: {'exo_in': indices['exo_in'], 'local_in':
                                                                indices['local_in'], 'global_in': indices['global_in'],
                                                                'global_out': indices['global_out'],
                                                                'surrogate': surr, 'is_computed': False}})

    def insert_component(self, component, exo_add=None, qoi_add=None):
        """Insert a new component into the system
        :param component: dict() specifying component model and connections (see docs of __init__ for details)
        :param exo_add: list() of BaseRVs to add to system exogenous inputs (will be appended to end)
        :param qoi_add: list() of UniformRVs specifying bounds of system output QoIs to add
        """
        if exo_add is not None:
            self.exo_vars.extend(exo_add)
        if qoi_add is not None:
            self.coupling_vars.extend(qoi_add)

        indices, surr = self.build_component(component)
        surr.init_coarse()
        self.graph.add_node(component['name'], exo_in=indices['exo_in'], local_in=indices['local_in'],
                            global_in=indices['global_in'], global_out=indices['global_out'], surrogate=surr,
                            is_computed=False)
        # Add graph edges
        neighbors = list(component['local_in'].keys())
        for neighbor in neighbors:
            self.graph.add_edge(neighbor, component['name'])
        self.logger.info(f"Inserted component '{component['name']}'.")

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
    def init_system(self):
        """Add the coarsest multi-index to each component surrogate"""
        self.print_title_str('Initializing all component surrogates')
        for node, node_obj in self.graph.nodes.items():
            node_obj['surrogate'].init_coarse()
            # for alpha, beta in list(node_obj['surrogate'].candidate_set):
            #     # Add one refinement in each input dimension to initialize
            #     node_obj['surrogate'].activate_index(alpha, beta)
            self.logger.info(f"Initialized component '{node}'.")

    @save_on_error
    def build_system(self, qoi_ind=None, N_refine=100, max_iter=20, max_tol=1e-3, max_runtime=1, save_interval=0,
                     prune_tol=1e-10, update_bounds=True, test_set=None, n_jobs=-1):
        """Build the system surrogate by iterative refinement until an end condition is met
        :param qoi_ind: list(), Indices of system QoI to focus refinement on, use all QoI if not specified
        :param N_refine: number of samples of exogenous inputs to compute error indicators on
        :param max_iter: the maximum number of refinement steps to take
        :param max_tol: the max allowable value in relative L2 error to achieve
        :param max_runtime: the maximum wall clock time (hr) to run refinement for (will go until all models finish)
        :param save_interval (int) number of refinement steps between each progress save, none if 0
        :param prune_tol: numerical tol in rel L2 error below which a cand multi-index is removed from consideration
        :param update_bounds: whether to continuously update coupling variable bounds during refinement
        :param test_set: a dict() of xt=(Nt, xdim), yt=(Nt, ydim) to show convergence of surrogate to the truth model
        :param n_jobs: number of cpu workers for computing error indicators (on master MPI task), 1=sequential
        """
        max_iter = self.refine_level + max_iter
        curr_error = np.inf
        t_start = time.time()

        # Record of (error indicator, component, alpha, beta, num_evals, total added cost (s)) for each iteration
        train_record = self.build_metrics.get('train_record', [])

        # Track convergence progress on a test set and on the max error indicator
        err_fig, err_ax = plt.subplots()
        test_stats, xt, yt, t_fig, t_ax, Nqoi = None, None, None, None, None, 0
        if test_set is not None:
            xt = test_set['xt']
            yt = test_set['yt']
            self.build_metrics['xt'] = xt
            self.build_metrics['yt'] = yt
            if self.build_metrics.get('test_stats') is not None:
                test_stats = self.build_metrics.get('test_stats')
            else:
                # Get initial perf metrics, (2, Nqoi)
                test_stats = np.expand_dims(self.get_test_metrics(xt, yt, qoi_ind=qoi_ind), axis=0)
            Nqoi = test_stats.shape[-1]
            t_fig, t_ax = plt.subplots(1, Nqoi) if Nqoi > 1 else plt.subplots()

        # Set up a parallel pool of workers, sequential if n_jobs=1
        with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
            while True:
                # Check all end conditions
                if self.refine_level >= max_iter:
                    self.print_title_str(f'Termination criteria reached: Max iteration {self.refine_level}/{max_iter}')
                    break
                if curr_error == -np.inf:
                    self.print_title_str(f'Termination criteria reached: No candidates left to refine')
                    break
                if curr_error < max_tol:
                    self.print_title_str(f'Termination criteria reached: L2 error {curr_error} < tol {max_tol}')
                    break
                if ((time.time() - t_start)/3600.0) >= max_runtime:
                    actual = datetime.timedelta(seconds=time.time()-t_start)
                    target = datetime.timedelta(seconds=max_runtime*3600)
                    self.print_title_str(f'Termination criteria reached: runtime {str(actual)} > {str(target)}')
                    break

                # Refine surrogate and save progress
                refine_res = self.refine(qoi_ind=qoi_ind, N_refine=N_refine, update_bounds=update_bounds,
                                         prune_tol=prune_tol, ppool=ppool)
                curr_error = refine_res[0]
                if save_interval > 0 and self.refine_level % save_interval == 0:
                    self.save_to_file(f'sys_iter_{self.refine_level}.pkl')

                # Plot progress of error indicator
                train_record.append(refine_res)
                error_record = [res[0] for res in train_record]
                self.build_metrics['train_record'] = train_record
                err_ax.clear(); err_ax.grid(); err_ax.plot(error_record, '-k')
                ax_default(err_ax, 'Iteration', 'Relative L2 error indicator', legend=False)
                err_ax.set_yscale('log')
                err_fig.savefig(str(Path(self.root_dir) / 'error_indicator.png'), dpi=300, format='png')

                # Plot progress on test set
                if test_set is not None:
                    stats = self.get_test_metrics(xt, yt, qoi_ind=qoi_ind)
                    test_stats = np.concatenate((test_stats, stats[np.newaxis, ...]), axis=0)
                    ind_use = qoi_ind if qoi_ind is not None else [int(i) for i in np.arange(Nqoi)]
                    for i in range(Nqoi):
                        ax = t_ax if Nqoi == 1 else t_ax[i]
                        ax.clear(); ax.grid(); ax.set_yscale('log')
                        ax.plot(test_stats[:, 1, i], '-k')
                        ax.set_title(self.coupling_vars[ind_use[i]].to_tex(units=True))
                        ax_default(ax, 'Iteration', r'Relative $L_2$ error', legend=False)
                    t_fig.set_size_inches(3.5*Nqoi, 3.5)
                    t_fig.tight_layout()
                    t_fig.savefig(str(Path(self.root_dir) / 'test_set.png'), dpi=300, format='png')
                    self.build_metrics['test_stats'] = test_stats

        self.save_to_file(f'sys_final.pkl')
        self.logger.info(f'Final system surrogate:\n {self}')

    def get_allocation(self, idx=None):
        """Get a breakdown of cost allocation up to a certain iteration number during training (starting at 1)
        :param idx: the iteration number to get allocation results for (defaults to last refinement step)
        :returns cost_alloc, offline_alloc, cost_cum: the cost alloc per node/fidelity and cumulative training cost
        """
        if idx is None:
            idx = self.refine_level
        if idx > self.refine_level:
            raise ValueError(f'Specified index: {idx} is greater than the max training level of {self.refine_level}')

        cost_alloc = dict()     # Cost allocation per node and model fidelity
        cost_cum = [0.0]          # Cumulative cost allocation during training

        # Add initialization costs for each node
        for node, node_obj in self.graph.nodes.items():
            surr = node_obj['surrogate']
            base_alpha = (0,) * len(surr.truth_alpha)
            base_beta = (0,) * (len(surr.max_refine) - len(surr.truth_alpha))
            base_cost = surr.get_cost(base_alpha, base_beta)
            cost_alloc[node] = dict()
            if base_cost > 0:
                cost_alloc[node][str(base_alpha)] = np.array([1, float(base_cost)])
                cost_cum[0] += float(base_cost)

        # Add cumulative training costs
        for i in range(idx):
            err_indicator, node, alpha, beta, num_evals, cost = self.build_metrics['train_record'][i]
            if cost_alloc[node].get(str(alpha), None) is None:
                cost_alloc[node][str(alpha)] = np.zeros(2)  # (num model evals, total cpu_time cost)
            cost_alloc[node][str(alpha)] += [round(num_evals), float(cost)]
            cost_cum.append(float(cost))

        # Get summary of total offline costs spent building search candidates (i.e. training overhead)
        offline_alloc = dict()
        for node, node_obj in self.graph.nodes.items():
            surr = node_obj['surrogate']
            offline_alloc[node] = dict()
            for alpha, beta in surr.candidate_set:
                if offline_alloc[node].get(str(alpha), None) is None:
                    offline_alloc[node][str(alpha)] = np.zeros(2)   # (num model evals, total cpu_time cost)
                added_cost = surr.get_cost(alpha, beta)
                base_cost = surr.get_sub_surrogate(alpha, beta).model_cost
                offline_alloc[node][str(alpha)] += [round(added_cost/base_cost), float(added_cost)]

        return cost_alloc, offline_alloc, np.cumsum(cost_cum)

    def get_test_metrics(self, xt, yt, qoi_ind=None, training=True):
        """Get relative L2 error metric over a test set
        :param xt: (Nt, xdim) random test set of inputs
        :param yt: (Nt, ydim) random test set outputs
        :param qoi_ind: list() of indices of QoIs to get metrics for
        :param training: whether to evaluate the surrogate in training or evaluation mode
        :returns stats: (2, Nqoi) -> [num_cands, Relative L2] for each QoI
        """
        if qoi_ind is None:
            qoi_ind = slice(None)
        ysurr = self(xt, training=training)
        ysurr = ysurr[:, qoi_ind]
        yt = yt[:, qoi_ind]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_l2_err = np.sqrt(np.mean((yt - ysurr) ** 2, axis=0)) / np.sqrt(np.mean(yt ** 2, axis=0))
            rel_l2_err = np.nan_to_num(rel_l2_err, posinf=np.nan, neginf=np.nan, nan=np.nan)
        num_cands = 0
        for node, node_obj in self.graph.nodes.items():
            num_cands += len(node_obj['surrogate'].index_set) + len(node_obj['surrogate'].candidate_set)

        # Get stats for each QoI
        stats = np.zeros((2, yt.shape[-1]))
        self.logger.debug(f'{"QoI idx":>10} {"Iteration":>10} {"len(I_k)":>10} {"Relative L2":>15}')
        for i in range(yt.shape[-1]):
            stats[:, i] = np.array([num_cands, rel_l2_err[i]])
            self.logger.debug(f'{i: 10d} {self.refine_level: 10d} {num_cands: 10d} {rel_l2_err[i]: 15.5f}')
        return stats

    def refine(self, qoi_ind=None, N_refine=100, update_bounds=True, prune_tol=1e-10, ppool=None):
        """Find and refine the component surrogate with the largest error on system QoI
        :param qoi_ind: list(), Indices of system QoI to focus surrogate refinement on, use all QoI if not specified
        :param N_refine: number of samples of exogenous inputs to compute error indicators on
        :param update_bounds: whether to continuously update coupling variable bounds
        :param prune_tol: set this for tolerance in NRMSE below which a candidate is no longer considered; NRMSE is a
                          normalized value and is independent of scale; prune_tol essentially acts as a numerical
                          tolerance for indicating when a candidate surrogate multi-index has negligible effect on
                          improving system QoI prediction
        :param ppool: a Parallel pool instance from joblib to compute error indicators in parallel, otherwise sequential
        :returns refine_res: a tuple() of (error_indicator, component, node_star, alpha_star, beta_star, N, cost)
                             indicating the chosen candidate index and incurred cost
        """
        self.print_title_str(f'Refining system surrogate: iteration {self.refine_level+1}')
        set_loky_pickler('dill')    # Dill can serialize 'self' for parallel workers
        temp_exc = self.executor    # It can't serialize an executor though, so must save this temporarily
        self.set_executor(None)
        if qoi_ind is None:
            qoi_ind = slice(None)

        # Compute entire integrated-surrogate on a random test set for global system QoI error estimation
        x_exo = self.sample_inputs((N_refine,))
        y_curr = self(x_exo, training=True)
        y_min, y_max = None, None
        if update_bounds:
            y_min = np.min(y_curr, axis=0, keepdims=True)  # (1, ydim)
            y_max = np.max(y_curr, axis=0, keepdims=True)  # (1, ydim)

        # Find the candidate surrogate with the largest error indicator
        error_max, error_indicator = -np.inf, -np.inf
        node_star, alpha_star, beta_star, l2_star, cost_star = None, None, None, -np.inf, 0
        for node, node_obj in self.graph.nodes.items():
            self.logger.info(f"Estimating error for component '{node}'...")
            candidates = node_obj['surrogate'].candidate_set.copy()

            def compute_error(alpha, beta):
                # Helper function for computing error indicators for a given candidate (alpha, beta)
                index_set = node_obj['surrogate'].index_set.copy()
                index_set.append((alpha, beta))
                y_cand = self(x_exo, training=True, index_set={node: index_set})
                ymin = np.min(y_cand, axis=0, keepdims=True)
                ymax = np.max(y_cand, axis=0, keepdims=True)
                error = y_cand[:, qoi_ind] - y_curr[:, qoi_ind]
                rel_l2 = np.sqrt(np.nanmean(error ** 2, axis=0)) / np.sqrt(np.nanmean(y_curr[:, qoi_ind] ** 2, axis=0))
                rel_l2 = np.nan_to_num(rel_l2, nan=np.nan, posinf=np.nan, neginf=np.nan)
                delta_error = np.nanmax(rel_l2)  # Max relative L2 error over all system QoIs
                delta_work = max(1, node_obj['surrogate'].get_cost(alpha, beta))  # Cpu time (s)

                return ymin, ymax, delta_error, delta_work

            if len(candidates) > 0:
                ret = ppool(delayed(compute_error)(alpha, beta) for alpha, beta in candidates) if ppool is not None \
                    else [compute_error(alpha, beta) for alpha, beta in candidates]

                for i, (ymin, ymax, d_error, d_work) in enumerate(ret):
                    if update_bounds:
                        y_min = np.min(np.concatenate((y_min, ymin), axis=0), axis=0, keepdims=True)
                        y_max = np.max(np.concatenate((y_max, ymax), axis=0), axis=0, keepdims=True)
                    alpha, beta = candidates[i]
                    error_indicator = d_error / d_work
                    if d_error > prune_tol:
                        self.logger.info(f"Candidate multi-index: {(alpha, beta)}. Error indicator: "
                                         f"{error_indicator}. L2 error: {d_error}")
                    else:
                        node_obj['surrogate'].prune_index(alpha, beta)
                        self.logger.info(f"PRUNED candidate multi-index: {(alpha, beta)}, since max relative L2 error "
                                         f"{d_error} < tol {prune_tol}")
                        continue

                    if error_indicator > error_max:
                        error_max = error_indicator
                        node_star, alpha_star, beta_star, l2_star, cost_star = node, alpha, beta, d_error, d_work
            else:
                self.logger.info(f"Component '{node}' has no available candidates left!")

        # Update all coupling variable ranges
        if update_bounds:
            for i in range(y_curr.shape[-1]):
                self.update_coupling_bds(i, (y_min[0, i], y_max[0, i]))

        # Add the chosen multi-index to the chosen component
        self.set_executor(temp_exc)
        if node_star is not None:
            self.logger.info(f"Candidate multi-index {(alpha_star, beta_star)} chosen for component '{node_star}'")
            self.graph.nodes[node_star]['surrogate'].activate_index(alpha_star, beta_star)
            self.refine_level += 1
            num_evals = round(cost_star / self[node_star].get_sub_surrogate(alpha_star, beta_star).model_cost)
        else:
            self.logger.info(f"No candidates left for refinement, iteration: {self.refine_level}")
            num_evals = 0

        return l2_star, node_star, alpha_star, beta_star, num_evals, cost_star

    def __call__(self, x, max_fpi_iter=100, anderson_mem=10, fpi_tol=1e-10, ground_truth=False, verbose=False,
                 training=False, index_set=None, qois=None):
        """Evaluate the system surrogate at exogenous inputs x
        :param x: (..., xdim) the points to be interpolated
        :param max_fpi_iter: the limit on convergence for the fixed-point iteration routine
        :param anderson_mem: hyperparameter for tuning the convergence of FPI with anderson acceleration
        :param fpi_tol: tolerance limit for convergence of fixed-point iteration
        :param ground_truth: whether to evaluate with the surrogates or the highest-fidelity 'ground truth' model
        :param verbose: whether to print out iteration progress during execution
        :param training: whether to call the system surrogate in training or evaluation mode
        :param index_set: dict() of {node:[indices]} to override default index set for a node (only useful for parallel)
        :param qois: list of qoi str ids or indices to return, defaults to returning all system qois
        :returns y: (..., ydim) the surrogate approximation of the system QoIs
        """
        # Allocate space for all system outputs (just save all coupling vars)
        xdim = x.shape[-1]
        ydim = len(self.coupling_vars)
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
                # for comp_name in node_obj['local_in']:
                #     assert self.graph.nodes[comp_name]['is_computed']
                coupling_inputs = y[..., node_obj['global_in']]
                comp_input = np.concatenate((exo_inputs, coupling_inputs), axis=-1)  # (..., xdim)

                # Compute outputs
                indices = index_set.get(scc[0], None) if index_set is not None else None
                comp_output = node_obj['surrogate'](comp_input[valid_idx, :], ground_truth=ground_truth,
                                                    training=training, index_set=indices)
                for local_i, global_i in enumerate(node_obj['global_out']):
                    y[valid_idx, global_i] = comp_output[..., local_i]
                node_obj['is_computed'] = True

                if verbose:
                    self.logger.info(f"Component '{scc[0]}' completed. Runtime: {time.time() - t1} s")

            # Handle FPI for SCCs with more than one component
            else:
                # Set the initial guess for all coupling vars (middle of domain)
                coupling_bds = [rv.bounds() for rv in self.coupling_vars]
                x_couple = np.array([(bds[0] + bds[1]) / 2 for bds in coupling_bds])
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
                            # assert self.graph.nodes[comp_name]['is_computed']
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
                        indices = index_set.get(node, None) if index_set is not None else None
                        comp_output = node_obj['surrogate'](comp_input[valid_idx, :], ground_truth=ground_truth,
                                                            training=training, index_set=indices)
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

        # Choose which qois to return, defaults to all
        if qois is None:
            qois = slice(None)
        else:
            if isinstance(qois[0], str):
                qois = [self.coupling_vars.index(str_id) for str_id in qois]

        # Return all component outputs (..., ydim), samples that didn't converge during FPI are left as np.nan
        return y[..., qois]

    def estimate_coupling_bds(self, N_est, anderson_mem=10, fpi_tol=1e-10, max_fpi_iter=100):
        """Estimate and set the coupling variable bounds
        :param N_est: The number of samples of exogenous inputs to use
        """
        self.print_title_str('Estimating coupling variable bounds')
        x = self.sample_inputs((N_est,))
        y = self(x, ground_truth=True, verbose=True, anderson_mem=anderson_mem, fpi_tol=fpi_tol,
                 max_fpi_iter=max_fpi_iter)
        for i in range(len(self.coupling_vars)):
            lb = np.nanmin(y[:, i])
            ub = np.nanmax(y[:, i])
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
        coupling_bds = [rv.bounds() for rv in self.coupling_vars]
        new_bds = offset_bds if init else (min(coupling_bds[global_idx][0], offset_bds[0]),
                                           max(coupling_bds[global_idx][1], offset_bds[1]))
        self.coupling_vars[global_idx].update_bounds(*new_bds)

        # Iterate over all components and update internal coupling variable bounds
        for node_name, node_obj in self.graph.nodes.items():
            if global_idx in node_obj['global_in']:
                # Get the local index for this coupling variable within each component's inputs
                local_idx = len(node_obj['exo_in']) + node_obj['global_in'].index(global_idx)
                node_obj['surrogate'].update_input_bds(local_idx, new_bds)

    def sample_inputs(self, shape, comp='System', use_pdf=False, nominal=None, constants=None):
        """Return samples of the inputs according to provided options
        :param shape: tuple() specifying shape of the samples
        :param comp: str() which component to sample inputs for (defaults to full system exogenous inputs)
        :param use_pdf: whether to sample from each variable's pdf, defaults to random samples over input domain instead
        :param nominal: dict() of {var_id: value} of nominal values for params with relative uncertainty, also can use
                        to specify constant values for a variable in constants
        :param constants: set() of param types to hold constant while sampling (i.e. calibration, design, etc.),
                          can also put a var_id string in here to specify a single variable to hold constant
        :returns x: (*shape, xdim) samples of the inputs
        """
        if nominal is None:
            nominal = dict()
        if constants is None:
            constants = set()
        vars = self.exo_vars if comp == 'System' else self[comp].x_vars
        xdim = len(vars)
        x = np.empty((*shape, xdim))
        for i, var in enumerate(vars):
            # Set a constant value for this variable
            if var.param_type in constants or str(var) in constants:
                x[..., i] = nominal.get(str(var), var.nominal)  # Defaults to variable's nominal value if not specified

            # Sample from this variable's pdf or randomly within its domain bounds
            else:
                x[..., i] = var.sample(shape, nominal=nominal.get(str(var), None)) if use_pdf \
                    else var.sample_domain(shape)

        return x

    def plot_slice(self, slice_idx, qoi_idx, compare_truth=False, N=50, nominal=None, random=False):
        """Helper function to plot 1d slices over the inputs (all other inputs set to nominal)
        :param slice_idx: list of exogenous input variables or indices to take 1d slices of
        :param qoi_idx: list of model output variables or indices to plot 1d slices of
        :param compare_truth: whether to also plot the ground truth model against the surrogate
        :param N: the number of points to take the 1d slice
        :param nominal: dict() of str(var)->nominal value to use as constant value for all non-sliced variables
        :param random: whether to slice in a random d-dimensional direction or hold all params constant while slicing
        :returns fig, ax with num_slice by num_qoi subplots
        """
        if nominal is None:
            nominal = dict()
        if isinstance(slice_idx[0], str):
            slice_idx = [self.exo_vars.index(str(var)) for var in slice_idx]
        if isinstance(qoi_idx[0], str):
            qoi_idx = [self.coupling_vars.index(str(var)) for var in qoi_idx]

        exo_bds = [var.bounds() for var in self.exo_vars]
        ub = [bds[1] for bds in exo_bds]
        xlabels = [self.exo_vars[idx].to_tex(units=True) for idx in slice_idx]
        ylabels = [self.coupling_vars[idx].to_tex(units=True) for idx in qoi_idx]

        xs = np.zeros((N, len(slice_idx), len(self.exo_vars)))
        for i in range(len(slice_idx)):
            xi_slice = np.linspace(exo_bds[slice_idx[i]][0], exo_bds[slice_idx[i]][1], N)

            if random:
                # Make a random walk across d-cube (threshold at domain maximums)
                vec = np.random.rand(len(self.exo_vars))
                vhat = vec / np.linalg.norm(vec)            # Random "positive" unit vector in d dimensions
                dxi = xi_slice[1] - xi_slice[0]             # Increment in the slice variable
                M = dxi / vhat[slice_idx[i]]                # Step size of the walk (or slice)
                dx = vhat * M                               # Increment for all directions
                xs[0, i, :] = np.array([bds[0] for bds in exo_bds])     # Start at lower left corner of domain
                for k in range(1, N):
                    xs[k, i, :] = np.minimum(xs[k-1, i, :] + dx, ub)
            else:
                # Otherwise, only slice one variable
                for j in range(len(self.exo_vars)):
                    if j == slice_idx[i]:
                        xs[:, i, j] = xi_slice              # 1d slice of input param of interest
                    else:
                        xs[:, i, j] = nominal.get(str(self.exo_vars[j]), self.exo_vars[j].nominal)

        if compare_truth:
            ys_model = self(xs, ground_truth=True)
        ys_surr = self(xs)

        # Make len(qoi) by len(inputs) grid of subplots
        fig, axs = plt.subplots(len(qoi_idx), len(slice_idx), sharex='col', sharey='row')
        for i in range(len(qoi_idx)):
            for j in range(len(slice_idx)):
                if len(qoi_idx) == 1:
                    ax = axs if len(slice_idx) == 1 else axs[j]
                elif len(slice_idx) == 1:
                    ax = axs if len(qoi_idx) == 1 else axs[i]
                else:
                    ax = axs[i, j]
                x = xs[:, j, slice_idx[j]]
                y_surr = ys_surr[:, j, qoi_idx[i]]
                if compare_truth:
                    y_model = ys_model[:, j, qoi_idx[i]]
                    ax.plot(x, y_model, '-k', label='Model')
                ax.plot(x, y_surr, '--r', label='Surrogate')
                ylabel = ylabels[i] if j == 0 else ''
                xlabel = xlabels[j] if i == len(qoi_idx) - 1 else ''
                legend = (i == 0 and j == len(slice_idx) - 1) and compare_truth
                ax_default(ax, xlabel, ylabel, legend=legend)
        fig.set_size_inches(3 * len(slice_idx), 3 * len(qoi_idx))
        fig.tight_layout()
        fig.savefig(Path(self.root_dir) / 'sweep.png', dpi=300, format='png')
        plt.show()
        return fig, axs

    def get_component(self, comp_name):
        """Return the ComponentSurrogate object for this component
        :param comp_name: (str) name of the component to return
        """
        return self.graph.nodes[comp_name]['surrogate']

    def print_title_str(self, title_str):
        """Quick wrapper to log an important message"""
        self.logger.info('-' * int(len(title_str)/2) + title_str + '-' * int(len(title_str)/2))

    def save_to_file(self, filename, save_dir=None):
        """Save the SystemSurrogate object to a .pkl file
        :param filename: filename of the .pkl file to save to
        :param save_dir: (str) Overrides existing save directory if provided
        """
        save_dir = save_dir if save_dir is not None else str(Path(self.root_dir) / 'sys')
        if save_dir is not None:
            if not Path(save_dir).is_dir():
                save_dir = '.'

            exec_temp = self.executor   # Temporarily save executor obj (can't pickle it)
            self.set_executor(None)
            with open(Path(save_dir) / filename, 'wb') as dill_file:
                dill.dump(self, dill_file)
            self.set_executor(exec_temp)
            self.logger.info(f'SystemSurrogate saved to {(Path(save_dir) / filename).resolve()}')

    def set_output_dir(self, set_dict):
        """Set the output directory for each component in set_dict
        :param set_dict: a dict() of component names (str) to their new output directories to set
        """
        for node, node_obj in self.graph.nodes.items():
            if node in set_dict:
                node_obj['surrogate'].set_output_dir(set_dict.get(node))

    def set_root_directory(self, dir, stdout=True):
        """Set the root to a new directory, for example if you move to a new system
        :param dir: str() or Path specifying new root directory
        :param stdout: whether to connect the logger to console (default)
        """
        self.root_dir = str(Path(dir).resolve())
        log_file = None
        if not (Path(self.root_dir) / 'sys').is_dir():
            os.mkdir(Path(self.root_dir) / 'sys')
        if not (Path(self.root_dir) / 'components').is_dir():
            os.mkdir(Path(self.root_dir) / 'components')
        for f in os.listdir(self.root_dir):
            if f.endswith('.log'):
                log_file = str((Path(self.root_dir) / f).resolve())
                break
        if log_file is None:
            fname = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.') + 'UTC_sys.log'
            log_file = str((Path(self.root_dir) / fname).resolve())

        # Setup the log file
        self.log_file = log_file
        self.logger = get_logger(self.__class__.__name__, log_file=log_file, stdout=stdout)

        # Update model output directories
        for node, node_obj in self.graph.nodes.items():
            surr = node_obj['surrogate']
            surr.logger = get_logger(surr.__class__.__name__, log_file=log_file, stdout=stdout)
            surr.log_file = self.log_file
            if surr.save_enabled():
                output_dir = str((Path(self.root_dir) / 'components' / node).resolve())
                if not Path(output_dir).is_dir():
                    os.mkdir(output_dir)
                surr.set_output_dir(output_dir)

    def __getitem__(self, component):
        """Convenience method to get the ComponentSurrogate object from the system
        :param component: (str) name of the component to get
        """
        return self.get_component(component)

    def __repr__(self):
        s = f'----SystemSurrogate----\nAdjacency: \n{nx.to_numpy_array(self.graph, dtype=int)}\n' \
            f'Exogenous inputs: {[str(var) for var in self.exo_vars]}\n'
        for node, node_obj in self.graph.nodes.items():
            s += f'Component: {node}\n{node_obj["surrogate"]}'
        return s

    def __str__(self):
        return self.__repr__()

    def set_executor(self, executor):
        """Set a new concurrent.futures.Executor object"""
        self.executor = executor
        for node, node_obj in self.graph.nodes.items():
            node_obj['surrogate'].executor = executor

    @staticmethod
    def load_from_file(filename, root_dir=None, executor=None):
        """Load a SystemSurrogate object from file
        :param filename: .pkl file to load
        :param root_dir: str() or Path to a folder to use as the root directory, (uses file's parent dir by default)
        :param executor: a concurrent.futures.Executor object to set, clears if None
        """
        if root_dir is None:
            root_dir = Path(filename).parent.parent     # Assume root/sys/filename.pkl

        with open(filename, 'rb') as dill_file:
            sys_surr = dill.load(dill_file)
            sys_surr.set_executor(executor)
            sys_surr.set_root_directory(root_dir)
            sys_surr.logger.info(f'SystemSurrogate loaded from {Path(filename).resolve()}')

        return sys_surr

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

    def __init__(self, multi_index, x_vars, model, truth_alpha, max_alpha=None, max_beta=None,
                 log_file=None, executor=None, model_args=(), model_kwargs=None):
        """Construct the MISC surrogate from a multi-index
        :param multi_index: [((alpha1), (beta1)), ... ] List of concatenated multi-indices alpha, beta specifying
                            model and surrogate fidelity
        :param x_vars: [X1, X2, ...] list of BaseRV() objects specifying bounds/pdfs for each input x
        :param model: The function to approximate, callable as y = model(x, alpha, *args, **kwargs)
        :param truth_alpha: tuple() specifying the highest model fidelity indices necessary for a 'truth' comparison
        :param max_alpha: tuple(), the maximum model refinement indices to allow, defaults to (3,...)
        :param max_beta: tuple(), the maximum surrogate refinement level indices, defaults to (3,...) of len xdim
        :param log_file: str() or Path-like specifying a log file
        :param executor: An instance of a concurrent.futures.executor, use to add candidate indices in parallel
        :param model_args: tuple() optional args to pass when calling the model
        :param model_kwargs: dict() optional kwargs to pass when calling the model
        """
        self.logger = get_logger(self.__class__.__name__, log_file=log_file)
        self.log_file = log_file
        assert self.is_downward_closed(multi_index), 'Must be a downward closed set.'
        self.index_set = []         # The active index set for the MISC approximation
        self.candidate_set = []     # Candidate indices for refinement
        self.pruned_set = []        # Indices removed from candidate set due to low error indicators (<5th percentile)
        self._model = model
        self._model_args = model_args
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.truth_alpha = truth_alpha
        self.x_vars = x_vars
        self.ydim = None
        max_alpha = (3,)*len(truth_alpha) if max_alpha is None else max_alpha
        max_beta = (3,)*len(self.x_vars) if max_beta is None else max_beta
        self.max_refine = list(max_alpha + max_beta)    # Max refinement indices
        self.executor = executor
        self.training_flag = None       # Keep track of which MISC coeffs are active
        # (True=active set, False=active+candidate sets, None=Neither/unknown)

        # Initialize important tree-like structures
        self.surrogates = dict()        # Maps alphas -> betas -> surrogates
        self.costs = dict()             # Maps alphas -> betas -> wall clock run times
        self.misc_coeff = dict()        # Maps alphas -> betas -> MISC coefficients

        # Construct vectors of [0,1]^dim(alpha+beta)
        Nij = len(self.max_refine)
        self.ij = np.zeros((2 ** Nij, Nij), dtype=np.uint8)
        for i, ele in enumerate(itertools.product([0, 1], repeat=Nij)):
            self.ij[i, :] = ele

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
        if ele in self.index_set:
            self.logger.warning(f'Multi-index {ele} is already in the active index set. Ignoring...')
            return

        # Add all possible new candidates (distance of one unit vector away)
        ind = list(alpha + beta)
        new_candidates = []
        for i in range(len(ind)):
            ind_new = ind.copy()
            ind_new[i] += 1

            # Don't add if we surpass a refinement limit
            if np.any(np.array(ind_new) > np.array(self.max_refine)):
                continue

            # Don't add if this index has been pruned previously
            new_cand = (tuple(ind_new[:len(alpha)]), tuple(ind_new[len(alpha):]))
            if new_cand in self.pruned_set:
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
                new_candidates.append(new_cand)

        # Build an interpolant for each new candidate
        if self.executor is None:   # Sequential
            for a, b in new_candidates:
                self.add_surrogate(a, b)
        else:                       # Parallel
            temp_exc = self.executor
            self.executor = None
            for a, b in new_candidates:
                if str(a) not in self.surrogates:
                    self.surrogates[str(a)] = dict()
                    self.costs[str(a)] = dict()
                    self.misc_coeff[str(a)] = dict()
            self._parallel_add_candidates(new_candidates, temp_exc)
            self.executor = temp_exc

        # Move to the active index set
        if ele in self.candidate_set:
            self.candidate_set.remove(ele)
        self.index_set.append(ele)
        new_candidates = [cand for cand in new_candidates if cand not in self.candidate_set]
        self.candidate_set.extend(new_candidates)
        self.training_flag = None   # Makes sure misc coeffs get recomputed next time

    def prune_index(self, alpha, beta):
        """Remove a multi-index from the candidate set"""
        ele = (alpha, beta)
        if ele in self.candidate_set:
            self.candidate_set.remove(ele)
            self.pruned_set.append(ele)

    def add_surrogate(self, alpha, beta):
        """Build a BaseInterpolator object for a given alpha, beta index
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        """
        # Create a dictionary for each alpha model to store multiple surrogate fidelities (beta)
        if str(alpha) not in self.surrogates:
            self.surrogates[str(alpha)] = dict()
            self.costs[str(alpha)] = dict()
            self.misc_coeff[str(alpha)] = dict()

        # Create a new interpolator object for this multi-index (abstract method)
        if self.surrogates[str(alpha)].get(str(beta), None) is None:
            self.logger.info(f'Building interpolant for index {(alpha, beta)} ...')
            x_new_idx, x_new, interp = self.add_interpolator(alpha, beta)
            self.surrogates[str(alpha)][str(beta)] = interp
            cost = self._add_interpolator(x_new_idx, x_new, interp)  # Awkward, but needed to separate the model evals
            self.costs[str(alpha)][str(beta)] = cost
            if self.ydim is None:
                self.ydim = interp.ydim()

    def init_coarse(self):
        """Initialize the coarsest interpolation and add to active index set"""
        alpha = (0,) * len(self.truth_alpha)
        beta = (0,) * len(self.max_refine[len(self.truth_alpha):])
        self.activate_index(alpha, beta)

    def iterate_candidates(self):
        """Iterate candidate indices one by one into the active index set
        :yields alpha, beta: the multi-indices of the current candidate that has been moved to active set
        """
        for alpha, beta in list(self.candidate_set):
            # Temporarily add a candidate index to active set
            self.index_set.append((alpha, beta))
            yield alpha, beta
            del self.index_set[-1]

    def __call__(self, x, ground_truth=False, truth_dir=None, training=False, index_set=None):
        """Evaluate the surrogate at points x
        :param x: (..., xdim) the points to be interpolated, must be within domain of x bounds
        :param ground_truth: whether to use the highest fidelity model or the surrogate (default)
        :param truth_dir: directory to save output files if ground_truth=True, ignored otherwise
        :param training: if True, then only compute with active index set, otherwise use all candidates as well
        :param index_set: a list() of (alpha, beta) to override self.index_set if given, else ignore
        :returns y: (..., ydim) the surrogate approximation of the qois
        """
        if ground_truth:
            # Bypass surrogate evaluation (don't save output)
            output_dir = self._model_kwargs.get('output_dir')
            if self.save_enabled():
                self._model_kwargs['output_dir'] = truth_dir

            ret = self._model(x, self.truth_alpha, *self._model_args, **self._model_kwargs)

            if output_dir is not None:
                self._model_kwargs['output_dir'] = output_dir

            return ret['y']

        # Decide which index set and corresponding misc coefficients to use
        misc_coeff = copy.deepcopy(self.misc_coeff)
        if index_set is None:
            # Use active indices + candidate indices depending on training mode
            index_set = self.index_set if training else self.index_set + self.candidate_set

            # Decide when to update misc coefficients
            if self.training_flag is None:
                misc_coeff = self.update_misc_coeffs(index_set)  # On initialization or reset
            else:
                if (not self.training_flag and training) or (self.training_flag and not training):
                    misc_coeff = self.update_misc_coeffs(index_set)  # Logical XOR cases for training mode

            # Save an indication of what state the MISC coefficients are in (i.e. training or eval mode)
            self.training_flag = training
        else:
            # If we passed in an index set, always recompute misc coeff and toggle for reset on next call
            misc_coeff = self.update_misc_coeffs(index_set)
            self.training_flag = None

        y = np.zeros(x.shape[:-1] + (self.ydim,))
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                func = self.surrogates[str(alpha)][str(beta)]
                y += comb_coeff * func(x)

        return y

    def update_misc_coeffs(self, index_set=None):
        """Update comb technique coeffs for MISC using the given index set, defaults to the active index set"""
        if index_set is None:
            index_set = self.index_set

        # Construct a (N_indices, dim(alpha+beta)) refactor of the index_set for arrayed computations
        index_mat = np.zeros((len(index_set), len(self.max_refine)), dtype=np.uint8)
        for i, (alpha, beta) in enumerate(index_set):
            index_mat[i, :] = alpha + beta
        index_mat = np.expand_dims(index_mat, axis=0)                               # (1, Ns, Nij)

        misc_coeff = dict()
        for alpha, beta in index_set:
            # Add permutations of [0, 1] to (alpha, beta)
            alpha_beta = np.array(alpha+beta, dtype=np.uint8)[np.newaxis, :]        # (1, Nij)
            new_indices = np.expand_dims(alpha_beta + self.ij, axis=1)              # (2**Nij, 1, Nij)

            # Find which indices are in the index_set (using np broadcasting comparison)
            diff = new_indices - index_mat                                  # (2**Nij, Ns, Nij)
            idx = np.count_nonzero(diff, axis=-1) == 0                      # (2**Nij, Ns)
            idx = np.any(idx, axis=-1)                                      # (2**Nij,)
            ij_use = self.ij[idx, :]                                        # (*, Nij)
            l1_norm = np.sum(np.abs(ij_use), axis=-1)                       # (*,)
            coeff = np.sum((-1) ** l1_norm)                                 # float

            # Save misc coeff to a dict() tree structure
            if misc_coeff.get(str(alpha)) is None:
                misc_coeff[str(alpha)] = dict()
            misc_coeff[str(alpha)][str(beta)] = coeff
            self.misc_coeff[str(alpha)][str(beta)] = coeff

        return misc_coeff

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
        try:
            return self.costs[str(alpha)][str(beta)]
        except:
            return 0

    def update_input_bds(self, idx, bds):
        """Update the bounds of the input at the given idx (assumes a uniform RV)
        :param idx: the index of the input variable to update
        :param bds: tuple() specifying the new bounds to update
        """
        self.x_vars[idx].update_bounds(*bds)

        # Update the bounds in all associated tensor-product surrogates
        for alpha in self.surrogates:
            for beta in self.surrogates[alpha]:
                self.surrogates[alpha][beta].update_input_bds(idx, bds)

    def save_enabled(self):
        """Return whether this model wants to save outputs to file"""
        return self._model_kwargs.get('output_dir') is not None

    def set_output_dir(self, output_dir):
        """Update the component model output directory
        :param output_dir: str() or Path specifying new directory for model output files
        """
        if output_dir is not None:
            output_dir = str(Path(output_dir).resolve())
        self._model_kwargs['output_dir'] = output_dir
        for alpha in self.surrogates:
            for beta in self.surrogates[alpha]:
                self.surrogates[alpha][beta]._model_kwargs['output_dir'] = output_dir

    def __repr__(self):
        s = f'Inputs \u2014 {[str(var) for var in self.x_vars]}\n'
        if self.training_flag is None:
            self.update_misc_coeffs()
            self.training_flag = True

        if self.training_flag:
            s += '(Training mode)\n'
            for alpha, beta in self.index_set:
                s += f"[{int(self.misc_coeff[str(alpha)][str(beta)])}] \u2014 {alpha}, {beta}\n"
            for alpha, beta in self.candidate_set:
                s += f"[-] \u2014 {alpha}, {beta}\n"
        else:
            s += '(Evaluation mode)\n'
            for alpha, beta in self.index_set + self.candidate_set:
                s += f"[{int(self.misc_coeff[str(alpha)][str(beta)])}] \u2014 {alpha}, {beta}\n"
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
        """Return a BaseInterpolator object and new refinement points for a given alpha, beta index
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        :returns x_new_idx, x_new, interp: list() of new grid indices, the new grid points (N_new, xdim), and the
                                           BaseInterpolator object. Similar to BaseInterpolator.refine()
        """
        pass

    @abstractmethod
    def _add_interpolator(self, x_new_idx, x_new, interp):
        """Secondary method to actually compute and save model evaluations within the interpolator
        :param x_new_idx: Return value of add_interpolator, list of new grid point indices
        :param x_new: (N_new, xdim), the new grid point locations
        :param interp: the BaseInterpolator object to compute model evaluations with
        :returns cost: the cost required to add this Interpolator object
        """
        pass

    @abstractmethod
    def _parallel_add_candidates(self, candidates, executor):
        """Defines a function to handle adding candidate indices in parallel. While add_interpolator() can make
        changes to 'self', these changes will not be saved in the master task if running in parallel over MPI.
        This method is a workaround so that all required mutable changes to 'self' are made in the master task, before
        distributing tasks to parallel workers using this method.
        :param candidates: list of [(alpha, beta),...] multi-indices
        :param executor: An instance of a concurrent.futures.Executor, use to iterate candidates in parallel
        """
        pass


class BaseInterpolator(ABC):
    """Base interpolator abstract class"""

    def __init__(self, beta, x_vars, xi=None, yi=None, model=None, model_args=(), model_kwargs=None):
        """Construct the interpolator
        :param beta: tuple(), refinement level indices for surrogate
        :param x_vars: list() of BaseRV() objects specifying bounds/pdfs for each input x
        :param xi: (Nx, xdim) interpolation points
        :param yi: the interpolation qoi values, y = (Nx, ydim)
        :param model: Callable as y = model(x), with x = (..., xdim), y = (..., ydim)
        :param model_args: tuple() of optional args for the model
        :param model_kwargs: dict() of optional kwargs for the model
        """
        self.logger = get_logger(self.__class__.__name__)
        self._model = model
        self._model_args = model_args
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.output_files = []                              # Save output files with same indexing as xi, yi
        self.xi = xi                                        # Interpolation points
        self.yi = yi                                        # Function values at interpolation points
        self.beta = beta                                    # Refinement level indices
        self.x_vars = x_vars                                # BaseRV() objects for each input
        self.model_cost = None                              # Total cpu time to evaluate model once (s)

    def update_input_bds(self, idx, bds):
        """Update the input bounds at the given index (assume a uniform RV)"""
        self.x_vars[idx].update_bounds(*bds)

    def xdim(self):
        """Get the dimension of the inputs"""
        return len(self.x_vars)

    def ydim(self):
        """Get the dimension of the outputs"""
        return self.yi.shape[-1] if self.yi is not None else None

    def save_enabled(self):
        """Return whether this model wants to save outputs to file"""
        return self._model_kwargs.get('output_dir') is not None

    def set_yi(self, yi=None, model=None, x_new=()):
        """Set the interpolation point qois, if yi is none then compute yi=model(self.xi)
        :param yi: (Nx, ydim) must match dimension of self.xi
        :param model: Callable as y, files = model(x), with x = (..., xdim), y = (..., ydim), files = list(.jsons),
                      model will only return y if output_dir is not passed as a kwarg
        :param x_new: tuple() specifying (x_new_idx, new_x), where new_x is an (N_new, xdim) array of new interpolation
                      points to include and x_new_idx is a list() specifying the indices of these points into self.xi,
                      overrides anything passed in for yi, and assumes a model is already specified
        :returns: dict() of {tuple(new_idx): np.array(yi)} if x_new_idx contains tuple elements, otherwise none
        """
        if model is not None:
            self._model = model
        if self._model is None:
            error_msg = 'Model not specified for computing QoIs at interpolation grid points.'
            self.logger.error(error_msg)
            raise Exception(error_msg)

        # Overrides anything passed in for yi (you would only be using this if yi was set previously)
        if x_new:
            new_idx = x_new[0]
            new_x = x_new[1]
            return_y = isinstance(new_idx[0], tuple)  # Return y rather than storing it if tuple indices are passed in
            ret = dict(y=dict(), files=dict())
            model_ret = self._model(new_x, *self._model_args, **self._model_kwargs)
            y_new, files_new, cpu_time = model_ret['y'], model_ret.get('files', None), model_ret.get('cost', 1)

            if self.save_enabled():
                for j in range(y_new.shape[0]):
                    if return_y:
                        ret['y'][str(new_idx[j])] = y_new[j, :].astype(np.float32)
                        ret['files'][str(new_idx[j])] = files_new[j]
                    else:
                        self.yi[new_idx[j], :] = y_new[j, :].astype(np.float32)
                        self.output_files[new_idx[j]] = files_new[j]
            else:
                for j in range(y_new.shape[0]):
                    if return_y:
                        ret['y'][str(new_idx[j])] = y_new[j, :].astype(np.float32)
                    else:
                        self.yi[new_idx[j], :] = y_new[j, :].astype(np.float32)

            if self.model_cost is None:
                self.model_cost = max(1, cpu_time)

            return ret

        # Set yi directly
        if yi is not None:
            self.yi = yi.astype(np.float32)
            return

        # Compute yi
        model_ret = self._model(self.xi, *self._model_args, **self._model_kwargs)
        self.yi, self.output_files, cpu_time = model_ret['y'], model_ret.get('files', list()), model_ret.get('cost', 1)

        if self.model_cost is None:
            self.model_cost = max(1, cpu_time)

    @abstractmethod
    def refine(self, beta, manual=False):
        """Return a new interpolator with one dimension refined by one level, specified by beta
        :param beta: list(), The new refinement level, should only refine one dimension
        :param manual: whether to manually compute model at refinement points
        :return interp: a refined BaseInterpolator object
             or x_new_idx, x_new, interp: where x_new are the newly refined interpolation points (N_new, xdim) and
                                          x_new_idx is the list of indices of these points into interp.xi and interp.yi,
                                          Would use this if you did not provide a callable model to the Interpolator or
                                          you want to manually set yi for each new xi outside this function or if you
                                          want to call set_yi() later with the new xi
        """
        pass

    @abstractmethod
    def __call__(self, x):
        """Evaluate the interpolation at points x
        :param x: (..., xdim) the points to be interpolated, must be within domain of self.xi
        :returns y: (..., ydim) the interpolated value of the qois
        """
        pass


class AnalyticalSurrogate(ComponentSurrogate):
    """Concrete 'surrogate' class that just uses the analytical model (i.e. bypasses surrogate evaluation)"""

    def __init__(self, multi_index, x_vars, model, truth_alpha, *args, **kwargs):
        """Initializes a stand-in ComponentSurrogate with all unnecessary fields set to empty. This ignores the
        multi_index and truth_alpha required args (since they don't mean anything for an analytical model).
        """
        kwargs['max_alpha'] = ()
        kwargs['max_beta'] = ()
        super().__init__([], x_vars, model, (), *args, **kwargs)

    # Override
    def __call__(self, x, **kwargs):
        """Evaluate the analytical model at points x, ignore extra kwargs passed in
        :param x: (..., xdim) the points to be evaluated
        :returns y: (..., ydim) the exact model output at the input points
        """
        ret = self._model(x, *self._model_args, **self._model_kwargs)
        return ret['y']

    # Override
    def activate_index(self, *args):
        """Do nothing"""
        pass

    # Override
    def add_surrogate(self, *args):
        """Do nothing"""
        pass

    # Override
    def init_coarse(self):
        """Do nothing"""
        pass

    # Override
    def update_misc_coeffs(self, **kwargs):
        """Do nothing"""
        pass

    # Override
    def get_sub_surrogate(self, *args):
        """Nothing to return for analytical model"""
        return None

    # Override
    def get_cost(self, *args):
        """Return nothing"""
        return 0

    def add_interpolator(self, *args):
        """Abstract method implementation, return none for an analytical model"""
        return None

    def _add_interpolator(self, *args):
        """Abstract method implementation, return cost=0 for an analytical model"""
        return 0

    def _parallel_add_candidates(self, *args):
        """Abstract method implementation, do nothing"""
        pass
