"""Module for sparse-grid surrogates"""
import numpy as np
from scipy.optimize import direct
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import itertools
import ast
import copy
import sys
from concurrent.futures import ALL_COMPLETED, wait

sys.path.append('..')

from surrogates.system import ComponentSurrogate, BaseInterpolator
from utils import get_logger, add_file_logging


class SparseGridSurrogate(ComponentSurrogate):
    """Concrete MISC surrogate class that maintains a sparse grid composed of smaller tensor-product grids"""

    def __init__(self, *args, **kwargs):
        """Construct a MISC surrogate that uses a sparse grid for interpolation, see the parent
        ComponentSurrogate class for other required args and kwargs.
        """
        # Initialize tree-like structures for maintaining a sparse grid of smaller tensor-product grids
        self.curr_max_beta = dict()     # Maps alphas -> current max refinement indices
        self.x_grids = dict()           # Maps alphas -> list of ndarrays specifying 1d grids corresponding to max_beta
        self.xi_map = dict()            # Maps grid locations/coords to interpolation points
        self.yi_map = dict()            # Maps grid locations/coords to interpolation qois
        self.yi_nan_map = dict()        # Maps grid locations to interpolated yi values when yi=nan
        self.yi_files = dict()          # Maps grid locations/coords to model output files (optional)
        super().__init__(*args, **kwargs)

    # Override super
    def __call__(self, x, ground_truth=False, training=False, index_set=None):
        """Evaluate the surrogate at points x (use xi,yi interpolation points specific to each sub tensor-product grid)
        :param x: (..., xdim) the points to be interpolated, must be within domain of x bounds
        :param ground_truth: whether to use the highest fidelity model or the surrogate (default)
        :param training: if True, then only compute with active index set, otherwise use all candidates as well
        :param index_set: a list() of (alpha, beta) to override self.index_set if given, else ignore
        :returns y: (..., ydim) the surrogate approximation of the qois
        """
        if ground_truth:
            # Bypass surrogate evaluation
            ret = self._model(x, self.truth_alpha, *self._model_args, **self._model_kwargs)
            y = ret[0] if self.save_enabled() else ret
            return y

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
                # Gather the xi/yi interpolation points/qois for this sub tensor-product grid
                interp = self.surrogates[str(alpha)][str(beta)]
                xi, yi = self.get_tensor_grid(alpha, beta)

                # Add this sub tensor-product grid to the MISC approximation
                y += comb_coeff * interp(x, xi=xi, yi=yi)

        return y

    def get_tensor_grid(self, alpha, beta, update_nan=True):
        """Convenience function to construct the xi/yi sub tensor-product grids for a given (alpha, beta) multi-index
        :param alpha: model fidelity multi-index, tuple()
        :param beta: surrogate fidelity multi-index, tuple()
        :param update_nan: try to fill nan with interpolated values, otherwise just return the nans in place
        :returns xi, yi: with shapes (prod(grid_sizes), xdim) and (prod(grid_sizes), ydim) respectively, the
                         interpolation grid points and corresponding qois for this tensor-product grid
        """
        interp = self.surrogates[str(alpha)][str(beta)]
        grid_sizes = interp.get_grid_sizes(beta)
        coords = [np.arange(grid_sizes[n]) for n in range(interp.xdim())]
        xi = np.zeros((np.prod(grid_sizes), interp.xdim()), dtype=np.float32)
        yi = np.zeros((np.prod(grid_sizes), self.ydim), dtype=np.float32)
        for i, coord in enumerate(itertools.product(*coords)):
            xi[i, :] = self.xi_map[str(alpha)][str(coord)]
            yi_curr = self.yi_map[str(alpha)][str(coord)]
            if update_nan and np.any(np.isnan(yi_curr)):
                # Try to replace NaN values if they are stored
                yi_curr = self.yi_nan_map[str(alpha)].get(str(coord), yi_curr)
            yi[i, :] = yi_curr

        return xi, yi

    def update_yi(self, alpha, beta, yi_dict):
        """Helper method to update yi values, accounting for possible nans"""
        self.yi_map[str(alpha)].update(yi_dict)
        lin_interp, nn_interp = None, None
        for grid_coord, yi in yi_dict.items():
            if np.any(np.isnan(yi)):
                lin_interp, nn_interp = self._update_yi_helper(alpha, beta, grid_coord, lin_interp, nn_interp)

        # Go back and try to re-interpolate old nan values as more points get added to the grid
        for grid_coord in list(self.yi_nan_map[str(alpha)].keys()):
            if grid_coord not in yi_dict:
                lin_interp, nn_interp = self._update_yi_helper(alpha, beta, grid_coord, lin_interp, nn_interp)

    def _update_yi_helper(self, alpha, beta, grid_coord, lin_interp, nn_interp):
        """Small helper function for building interpolants when updating yi"""
        if lin_interp is None or nn_interp is None:
            xi_mat, yi_mat = self.get_tensor_grid(alpha, beta, update_nan=False)
            nan_idx = np.any(np.isnan(yi_mat), axis=-1)
            xi_mat = xi_mat[~nan_idx, :]
            yi_mat = yi_mat[~nan_idx, :]
            try:
                lin_interp = LinearNDInterpolator(xi_mat, yi_mat, rescale=True)
            except:
                pass
            try:
                nn_interp = NearestNDInterpolator(xi_mat, yi_mat, rescale=True)
            except:
                pass

        x_interp = self.xi_map[str(alpha)][str(grid_coord)]  # (xdim,)
        yi = np.zeros((self.ydim,)) * np.nan
        try:
            yi = lin_interp(x_interp)
        except:
            pass  # If linear interpolation didn't work, just default to nearest neighbor
        if np.any(np.isnan(yi)):
            try:
                yi = nn_interp(x_interp)
            except:
                pass

        # If any nans are left, then something else is going wrong, and you need to fix this error
        if np.any(np.isnan(yi)):
            raise Exception(f'Trying to interpolate with NaNs for beta {beta}, please check model '
                            f'{self._model} for too many NaN outputs.')
        self.yi_nan_map[str(alpha)][str(grid_coord)] = np.atleast_1d(np.squeeze(yi))
        return lin_interp, nn_interp

    # Override
    def get_sub_surrogate(self, alpha, beta, include_grid=False):
        """Get the specific sub-surrogate corresponding to the (alpha,beta) fidelity
        :param alpha: A multi-index (tuple) specifying model fidelity
        :param beta: A multi-index (tuple) specifying surrogate fidelity
        :param include_grid: whether to add the xi/yi points to the returned BaseInterpolator object
        """
        interp = super().get_sub_surrogate(alpha, beta)
        if include_grid:
            interp.xi, interp.yi = self.get_tensor_grid(alpha, beta)
        return interp

    def add_interpolator(self, alpha, beta):
        """Abstract method implementation for constructing tensor-product grid interpolants"""
        # Create a new tensor-product grid interpolator for the base index (0, 0, ...)
        if np.sum(beta) == 0:
            args = (alpha,) + self._model_args
            interp = TensorProductInterpolator(beta, self.x_vars, model=self._model, model_args=args,
                                               model_kwargs=self._model_kwargs, init_grids=True, reduced=True)
            x_pt = np.array([float(interp.x_grids[n][beta[n]]) for n in range(interp.xdim())], dtype=np.float32)
            self.curr_max_beta[str(alpha)] = list(beta)
            self.x_grids[str(alpha)] = copy.deepcopy(interp.x_grids)
            self.xi_map[str(alpha)] = {str(beta): x_pt}
            self.yi_map[str(alpha)] = dict()
            self.yi_nan_map[str(alpha)] = dict()
            if self.save_enabled():
                self.yi_files[str(alpha)] = dict()

            return [beta], x_pt.reshape((1, len(self.x_vars))), interp
        # Otherwise, all other indices are a refinement of previous grids

        # Look for first multi-index neighbor that is one level of refinement away
        refine_tup = None
        for beta_old_str in list(self.surrogates[str(alpha)].keys()):
            beta_old = ast.literal_eval(beta_old_str)
            if self.is_one_level_refinement(beta_old, beta):
                idx_refine = int(np.nonzero(np.array(beta, dtype=int) - np.array(beta_old, dtype=int))[0])
                refine_level = beta[idx_refine]
                if refine_level > self.curr_max_beta[str(alpha)][idx_refine]:
                    # Generate next refinement grid and save (refine_tup = tuple(x_new_idx, x_new, interp))
                    refine_tup = self.surrogates[str(alpha)][beta_old_str].refine(beta, manual=True)
                    self.curr_max_beta[str(alpha)][idx_refine] = refine_level
                    self.x_grids[str(alpha)][idx_refine] = copy.deepcopy(refine_tup[2].x_grids[idx_refine])
                else:
                    # Access the refinement grid from memory (it is already computed)
                    num_pts = self.surrogates[str(alpha)][beta_old_str].get_grid_sizes(beta)[idx_refine]
                    x_refine = self.x_grids[str(alpha)][idx_refine][:num_pts]
                    refine_tup = self.surrogates[str(alpha)][beta_old_str].refine(beta, x_refine=x_refine,
                                                                                  manual=True)
                break  # Only need to grab one neighbor

        # Gather new interpolation grid points
        x_new_idx, x_new, interp = refine_tup
        xn_coord = []   # Save multi-index coordinates of points to compute model at for refinement
        xn_pts = np.zeros((0, interp.xdim()), dtype=np.float32)     # Save physical x location of new points
        for i, multi_idx in enumerate(x_new_idx):
            if str(multi_idx) not in self.yi_map[str(alpha)]:
                # We have not computed this grid coordinate yet
                xn_coord.append(multi_idx)
                xn_pts = np.concatenate((xn_pts, x_new[i, np.newaxis, :]), axis=0)  # (N_new, xdim)
                self.xi_map[str(alpha)][str(multi_idx)] = x_new[i, :]

        return xn_coord, xn_pts, interp

    def _add_interpolator(self, x_new_idx, x_new, interp):
        """Awkward solution, I know, but actually compute and save the model evaluations here"""
        # Compute and store model output at new refinement points in a hash structure
        yi_ret = interp.set_yi(x_new=(x_new_idx, x_new))
        alpha = interp._model_args[0]
        if self.save_enabled():
            self.update_yi(alpha, interp.beta, yi_ret[0])
            self.yi_files[str(alpha)].update(yi_ret[1])
        else:
            self.update_yi(alpha, interp.beta, yi_ret)
        cost = interp.wall_time * len(x_new_idx)

        if self.ydim is None:
            for coord_str, yi in self.yi_map[str(alpha)].items():
                self.ydim = yi.shape[0]
                break

        return cost

    def _parallel_add_candidates(self, candidates, executor):
        """Work-around to make sure mutable instance variable changes are made before/after
        splitting tasks using this method over parallel (potentially MPI) workers. MPI workers cannot save changes to
        'self', so this method should only distribute static tasks to the workers.
        :param candidates: list of [(alpha, beta),...] candidate multi-indices
        :param executor: An instance of a concurrent.futures.Executor, use to iterate candidates in parallel
        """
        # Do sequential tasks first (i.e. make mutable changes to self), build up parallel task args
        task_args = []
        for alpha, beta in candidates:
            x_new_idx, x_new, interp = self.add_interpolator(alpha, beta)
            task_args.append((alpha, beta, x_new_idx, x_new, interp))

        def parallel_task(alpha, beta, x_new_idx, x_new, interp):
            # Must return anything you want changed in self or interp (mutable changes aren't saved over MPI workers)
            logger = get_logger(__name__)
            logger_child = logger.getChild(self.__class__.__name__)
            if self.log_file is not None:
                add_file_logging(logger_child, self.log_file, suppress_stdout=True)
            logger_child.info(f'Building interpolant for index {(alpha, beta)} ...')
            yi_ret = interp.set_yi(x_new=(x_new_idx, x_new))
            wall_time = interp.wall_time if interp.wall_time is not None else 1
            return yi_ret, wall_time

        # Wait for all parallel workers to return
        fs = [executor.submit(parallel_task, *args) for args in task_args]
        wait(fs, timeout=None, return_when=ALL_COMPLETED)

        # Update self and interp with the results from all workers (and check for errors)
        for i, future in enumerate(fs):
            try:
                a = task_args[i][0]
                b = task_args[i][1]
                x_new_idx = task_args[i][2]
                interp = task_args[i][4]
                yi_ret = future.result()
                wall_time = yi_ret[1]
                interp.wall_time = wall_time
                self.surrogates[str(a)][str(b)] = interp
                if self.save_enabled():
                    self.update_yi(a, b, yi_ret[0][0])
                    self.yi_files[str(a)].update(yi_ret[0][1])
                else:
                    self.update_yi(a, b, yi_ret[0])
                self.costs[str(a)][str(b)] = interp.wall_time * len(x_new_idx)

                if self.ydim is None:
                    for coord_str, yi in self.yi_map[str(a)].items():
                        self.ydim = yi.shape[0]
                        break
            except:
                self.logger.error(f'An exception occurred in a thread handling add_interpolator{candidates[i]}')
                raise


class TensorProductInterpolator(BaseInterpolator):
    """Abstract tensor-product (multivariate) interpolator"""

    def __init__(self, beta, x_vars, init_grids=True, reduced=False, **kwargs):
        """Initialize a tensor-product grid interpolator
        :param beta: tuple(), refinement level in each dimension of xdim
        :param x_vars: list() of BaseRV() objects specifying bounds/pdfs for each input x
        :param init_grids: Whether to compute 1d leja sequences on init
        :param reduced: Whether to store xi/yi matrices, e.g. set true if storing in external sparse grid structure
        """
        self.weights = []   # Barycentric weights for each dimension
        self.x_grids = []   # Univariate nested leja sequences in each dimension
        self.reduced = reduced
        super().__init__(beta, x_vars, **kwargs)

        if init_grids:
            # Construct 1d univariate Leja sequences in each dimension
            grid_sizes = self.get_grid_sizes(self.beta)
            self.x_grids = [self.leja_1d(grid_sizes[n], self.x_vars[n].bounds(), wt_fcn=self.x_vars[n].pdf).astype(np.float32)
                            for n in range(self.xdim())]

            for n in range(self.xdim()):
                Nx = grid_sizes[n]
                bds = self.x_vars[n].bounds()
                grid = self.x_grids[n]
                C = (bds[1] - bds[0]) / 4.0  # Interval capacity (see Berrut and Trefethen 2004)
                xj = grid.reshape((Nx, 1))
                xi = grid.reshape((1, Nx))
                dist = (xj - xi) / C
                np.fill_diagonal(dist, 1)  # Ignore product when i==j
                self.weights.append((1.0 / np.prod(dist, axis=1)).astype(np.float32))  # (Nx,)

            # Cartesian product of univariate grids
            if not self.reduced:
                self.xi = np.zeros((np.prod(grid_sizes), self.xdim()), dtype=np.float32)
                for i, ele in enumerate(itertools.product(*self.x_grids)):
                    self.xi[i, :] = ele

    def refine(self, beta, manual=False, x_refine=None):
        """Return a new interpolator with one dimension refined by one level, specified by beta
        :param beta: tuple(), The new refinement level, should only refine one dimension
        :param manual: whether to manually compute model at refinement points
        :param x_refine: (Nx,) use this array as the refined 1d grid if provided, otherwise compute via leja_1d
        :return interp: a TensorProductInterpolator with a refined grid
             or x_new_idx, x_new, interp: where x_new are the newly refined interpolation points (N_new, xdim) and
                                          x_new_idx is the list of indices of these points into interp.xi and interp.yi,
                                          Would use this if you did not provide a callable model to the Interpolator or
                                          you want to manually set yi for each new xi outside this function. Sets
                                          elements of x_new_idx to tuple() indices if self.reduced, otherwise integers
        """
        # Initialize a new interpolant with the new refinement levels
        try:
            interp = TensorProductInterpolator(beta, self.x_vars, model=self._model, model_args=self._model_args,
                                               model_kwargs=self._model_kwargs, init_grids=False, reduced=self.reduced)

            # Find the dimension and number of new points to add
            old_grid_sizes = self.get_grid_sizes(self.beta)
            new_grid_sizes = interp.get_grid_sizes(beta)
            dim_refine = 0
            num_new_pts = 0
            for idx, grid_size in enumerate(new_grid_sizes):
                if grid_size != old_grid_sizes[idx]:
                    dim_refine = idx
                    num_new_pts = grid_size - old_grid_sizes[idx]
                    break

            # Add points to leja grid in this dimension
            interp.x_grids = copy.deepcopy(self.x_grids)
            xi = copy.deepcopy(x_refine) if x_refine is not None else self.leja_1d(num_new_pts,
                                                                                   interp.x_vars[dim_refine].bounds(),
                                                                                   z_pts=interp.x_grids[dim_refine],
                                                                                   wt_fcn=interp.x_vars[dim_refine].pdf)
            interp.x_grids[dim_refine] = xi.astype(np.float32)

            # Update barycentric weights in this dimension
            interp.weights = copy.deepcopy(self.weights)
            Nx_old = old_grid_sizes[dim_refine]
            Nx_new = new_grid_sizes[dim_refine]
            old_wts = copy.deepcopy(self.weights[dim_refine])
            new_wts = np.zeros(Nx_new, dtype=np.float32)
            new_wts[:Nx_old] = old_wts
            bds = interp.x_vars[dim_refine].bounds()
            C = (bds[1] - bds[0]) / 4.0  # Interval capacity
            xi = interp.x_grids[dim_refine]
            for j in range(Nx_old, Nx_new):
                new_wts[:j] *= (C / (xi[:j] - xi[j]))
                new_wts[j] = np.prod(C / (xi[j] - xi[:j]))
            interp.weights[dim_refine] = new_wts

            # Copy yi over at existing interpolation points
            x_new = np.zeros((0, interp.xdim()), dtype=np.float32)
            x_new_idx = []
            tol = 1e-12     # Tolerance for floating point comparison
            j = 0           # Use this idx for iterating over existing yi
            if not self.reduced:
                interp.xi = np.zeros((np.prod(new_grid_sizes), self.xdim()), dtype=np.float32)
                interp.yi = np.zeros((np.prod(new_grid_sizes), self.ydim()), dtype=np.float32)
                if self.save_enabled():
                    interp.output_files = [None] * np.prod(new_grid_sizes)

            old_indices = [np.arange(old_grid_sizes[n]) for n in range(self.xdim())]
            old_indices = list(itertools.product(*old_indices))
            new_indices = [np.arange(new_grid_sizes[n]) for n in range(self.xdim())]
            new_indices = list(itertools.product(*new_indices))
            for i in range(len(new_indices)):
                # Get the new grid coordinate/index and physical x location/point
                new_x_idx = new_indices[i]
                new_x_pt = np.array([float(interp.x_grids[n][new_x_idx[n]]) for n in range(self.xdim())],
                                    dtype=np.float32)

                if not self.reduced:
                    # Store the old xi/yi and return new x points
                    interp.xi[i, :] = new_x_pt
                    if j < len(old_indices) and np.all(np.abs(np.array(old_indices[j]) -
                                                              np.array(new_indices[i])) < tol):
                        # If we already have this interpolation point
                        interp.yi[i, :] = self.yi[j, :]
                        if self.save_enabled():
                            interp.output_files[i] = self.output_files[j]
                        j += 1
                    else:
                        # Otherwise, save new interpolation point and its index
                        x_new = np.concatenate((x_new, new_x_pt.reshape((1, self.xdim()))))
                        x_new_idx.append(i)
                else:
                    # Just find the new x indices and return those for the reduced case
                    if j < len(old_indices) and np.all(np.abs(np.array(old_indices[j]) -
                                                              np.array(new_indices[i])) < tol):
                        j += 1
                    else:
                        x_new = np.concatenate((x_new, new_x_pt.reshape((1, self.xdim()))))
                        x_new_idx.append(new_x_idx)     # Add a tuple() multi-index if not saving xi/yi

            # Evaluate the model at new interpolation points
            interp.wall_time = self.wall_time
            if self._model is None:
                self.logger.warning(f'No model available to evaluate new interpolation points, returning the points '
                                    f'to you instead...')
                return x_new_idx, x_new, interp
            elif manual or self.reduced:
                return x_new_idx, x_new, interp
            else:
                interp.set_yi(x_new=(x_new_idx, x_new))
                return interp

        except Exception as e:
            import traceback
            tb_str = str(traceback.format_exception(e))
            self.logger.error(tb_str)
            raise Exception(f'Original exception in refine(): {tb_str}')

    def __call__(self, x, xi=None, yi=None):
        """Evaluate the barycentric interpolation at points x (abstract implementation)
        :param x: (..., xdim) the points to be interpolated, must be within domain of self.xi
        :param xi: (Ni, xdim) optional, interpolation grid points to use (e.g. if self.xi is none)
        :param yi: (Ni, ydim) optional, interpolation qois at xi to use (e.g. if self.yi is none)
        :returns y: (..., ydim) the interpolated value of the qois
        """
        # Use linear/NN interpolation for yi=nan values (may have resulted from bad model outputs)
        if yi is None:
            yi = self.yi.copy()
        if xi is None:
            xi = self.xi.copy()
        ydim = yi.shape[-1]
        nan_idx = np.any(np.isnan(yi), axis=-1)
        if np.any(nan_idx):
            try:
                lin_interp = LinearNDInterpolator(xi[~nan_idx, :], yi[~nan_idx, :], rescale=True)
                yi[nan_idx, :] = lin_interp(xi[nan_idx, :])
            except:
                pass  # If linear interpolation didn't work, just default to nearest neighbor
            nan_idx = np.any(np.isnan(yi), axis=-1)
            if np.any(nan_idx):
                try:
                    nn_interp = NearestNDInterpolator(xi[~nan_idx, :], yi[~nan_idx, :], rescale=True)
                    yi[nan_idx, :] = nn_interp(xi[nan_idx, :])
                except:
                    pass

        # If any nans are left, then something else is going wrong, and you need to fix this error
        if np.any(np.isnan(yi)):
            raise Exception(f'Trying to interpolate with NaNs for beta {self.beta}, please check model {self._model} '
                            f'for too many NaN outputs.')

        # Loop over multi-indices and compute tensor-product lagrange polynomials
        grid_sizes = self.get_grid_sizes(self.beta)
        y = np.zeros(x.shape[:-1] + (ydim,))    # (..., ydim)
        indices = [np.arange(grid_sizes[n]) for n in range(self.xdim())]
        for i, j in enumerate(itertools.product(*indices)):
            L_j = np.empty(x.shape)             # (..., xdim)

            # Compute univariate Lagrange polynomials in each dimension
            for n in range(self.xdim()):
                x_n = x[..., n, np.newaxis]     # (..., 1)
                x_j = self.x_grids[n]           # (Nx,)
                w_j = self.weights[n]           # (Nx,)

                # Compute the jth Lagrange basis polynomial L_j(x_n) for this x dimension (in barycentric form)
                c = x_n - x_j
                div_zero_idx = np.abs(c) <= 1e-4 * x_j + 1e-8   # Track where x is approx at an interpolation pnt x_j
                c[div_zero_idx] = 1                             # Temporarily set to 1 to avoid divide by zero error
                c = w_j / c
                L_j[..., n] = c[..., j[n]] / np.sum(c, axis=-1)  # (...) same size as original x

                # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
                L_j[div_zero_idx[..., j[n]], n] = 1
                L_j[np.any(div_zero_idx[..., [idx for idx in range(grid_sizes[n]) if idx != j[n]]], axis=-1), n] = 0

            # Add multivariate basis polynomial contribution to interpolation output
            L_j = np.prod(L_j, axis=-1, keepdims=True)      # (..., 1)
            y += L_j * yi[i, :]

        return y

    @staticmethod
    def get_grid_sizes(beta, k=4):
        """Compute number of grid points in each dimension"""
        return [k*beta[i] + 1 for i in range(len(beta))]

    @staticmethod
    def leja_1d(N, z_bds, z_pts=None, wt_fcn=None):
        """Find the next N points in the Leja sequence of z_pts
        :param N: Number of new points to add to the sequence
        :param z_bds: Bounds on 1d domain (tuple)
        :param z_pts: Current univariate leja sequence (Nz,), start at middle of z_bds if None
        :param wt_fcn: Weighting function, uses a constant weight if None, callable as wt_fcn(z)
        """
        if wt_fcn is None:
            wt_fcn = lambda z: 1
        if z_pts is None:
            z_pts = (z_bds[1] + z_bds[0]) / 2
            N = N - 1
        z_pts = np.atleast_1d(z_pts).astype(np.float32)

        # Construct leja sequence by maximizing the objective sequentially
        for i in range(N):
            obj_fun = lambda z: -wt_fcn(np.array(z).astype(np.float32)) * np.prod(np.abs(z - z_pts))
            res = direct(obj_fun, [z_bds])  # Use global DIRECT optimization over 1d domain
            z_star = res.x
            z_pts = np.concatenate((z_pts, z_star))

        return z_pts
