"""Module for sparse-grid surrogates"""
import numpy as np
from scipy.optimize import direct
import itertools
import ast
import copy
import sys
from abc import abstractmethod

sys.path.append('..')

from surrogates.system import ComponentSurrogate, BaseInterpolator


class SparseGridSurrogate(ComponentSurrogate):
    """Concrete MISC surrogate class that maintains a sparse grid composed of smaller tensor-product grids"""

    def __init__(self, *args, interp='lagrange', **kwargs):
        """Construct a MISC surrogate that uses a sparse grid for interpolation, see the parent
        ComponentSurrogate class for other required args and kwargs.
        :param interp: (str) the interpolation method to use, defaults to barycentric Lagrange interpolation
        """
        # Initialize tree-like structures for maintaining a sparse grid of smaller tensor-product grids
        self.curr_max_beta = dict()     # Maps alphas -> current max refinement indices
        self.x_grids = dict()           # Maps alphas -> list of ndarrays specifying 1d grids corresponding to max_beta
        self.interp_class = None

        # Must be a class that implements TensorProductInterpolator
        match interp:
            case 'lagrange':
                self.interp_class = LagrangeInterpolator
            case other:
                raise NotImplementedError(f"Interpolation type '{interp}' is not known at this time.")

        super().__init__(*args, **kwargs)

    def add_interpolator(self, alpha, beta):
        """Abstract method implementation for constructing tensor-product grid interpolants"""
        # Create a new tensor-product grid interpolator for the base index (0, 0, ...)
        if np.sum(beta) == 0:
            interp = self.interp_class(list(beta), self.x_vars, model=self.alpha_models[str(alpha)])
            interp.set_yi()
            cost = interp.wall_time * interp.xi.shape[0]
            self.curr_max_beta[str(alpha)] = list(beta)
            self.x_grids[str(alpha)] = copy.deepcopy(interp.x_grids)
            return interp, cost
        # Otherwise, all other indices are a refinement of previous grids

        # Look for multi-index neighbors that are one level of refinement away
        neighbors = []
        for beta_old_str in self.surrogates[str(alpha)]:
            beta_old = ast.literal_eval(beta_old_str)
            if self.is_one_level_refinement(beta_old, beta):
                idx_refine = int(np.nonzero(np.array(beta, dtype=int) - np.array(beta_old, dtype=int))[0])
                refine_level = beta[idx_refine]
                if refine_level > self.curr_max_beta[str(alpha)][idx_refine]:
                    # Generate next refinement grid and save (refine_tup = tuple(x_new_idx, x_new, interp))
                    refine_tup = self.surrogates[str(alpha)][beta_old_str].refine(list(beta), manual=True)
                    self.curr_max_beta[str(alpha)][idx_refine] = refine_level
                    self.x_grids[str(alpha)][idx_refine] = copy.deepcopy(refine_tup[2].x_grids[idx_refine])
                    neighbors.append(refine_tup)
                else:
                    # Access the refinement grid from memory (it is already computed)
                    num_pts = self.surrogates[str(alpha)][beta_old_str].get_grid_sizes(beta)[idx_refine]
                    x_refine = self.x_grids[str(alpha)][idx_refine][:num_pts]
                    refine_tup = self.surrogates[str(alpha)][beta_old_str].refine(list(beta), x_refine=x_refine,
                                                                                  manual=True)
                    neighbors.append(refine_tup)

        if len(neighbors) > 1:
            xi = neighbors[0][2].xi
            for n in neighbors[1:]:
                diff = xi - n[2].xi
                if not np.isclose(np.max(diff), 0):
                    raise Exception("Buh oh, you just can't figure it out can ya")

        # Refine the neighbor with the fewest new points (and merge pre-computed points from other neighbors)
        min_idx = np.argmin([len(neighbor[0]) for neighbor in neighbors])
        x_new_idx, x_new, interp = neighbors.pop(min_idx)
        del_idx = []
        for neighbor in neighbors:
            for i, interp_idx in enumerate(x_new_idx):
                # If a neighbor's new index list doesn't have an index, then the neighbor has already computed it
                if interp_idx not in neighbor[0] and i not in del_idx:
                    # Assume the exact same interpolation indexing for all neighbors (true for tensor-product grids)
                    interp.yi[interp_idx, :] = neighbor[2].yi[interp_idx, :]
                    del_idx.append(i)
        x_new_idx = list(np.delete(x_new_idx, del_idx))
        x_new = np.delete(x_new, del_idx, axis=0)

        # Compute the model outputs at all remaining refinement points
        y_new = self.alpha_models[str(alpha)](x_new)  # (N_new, ydim)
        for j in range(y_new.shape[0]):
            interp.yi[x_new_idx[j], :] = y_new[j, :]

        cost = interp.wall_time * len(x_new_idx)
        return interp, cost


class TensorProductInterpolator(BaseInterpolator):
    """Abstract tensor-product (multivariate) interpolator"""

    def __init__(self, beta, x_vars, yi=None, model=None, init_grids=True):
        """Initialize a tensor-product grid interpolator
        :param beta: list(), refinement level in each dimension of xdim
        :param x_vars: list() of BaseRV() objects specifying bounds/pdfs for each input x
        :param yi: the interpolation qoi values, y = (prod(xdims), ydim)
        :param model: Callable as y = model(x), with x = (..., xdim), y = (..., ydim)
        :param init_grids: Whether to compute 1d leja sequences on init
        """
        self.x_grids = []   # Univariate leja sequences in each dimension
        super().__init__(beta, x_vars, yi=yi, model=model)

        if init_grids:
            # Construct 1d univariate Leja sequences in each dimension
            grid_sizes = self.get_grid_sizes(self.beta)
            self.x_grids = [self.leja_1d(grid_sizes[n], self.x_bds[n], wt_fcn=self.x_vars[n].pdf)
                            for n in range(self.xdim())]

            # Cartesian product of univariate grids
            self.xi = np.zeros((np.prod(grid_sizes), self.xdim()))
            for i, ele in enumerate(itertools.product(*self.x_grids)):
                self.xi[i, :] = ele

    def refine(self, beta, manual=False, x_refine=None):
        """Return a new interpolator with one dimension refined by one level, specified by beta
        :param beta: list(), The new refinement level, should only refine one dimension
        :param manual: whether to manually compute model at refinement points
        :param x_refine: (Nx,) use this array as the refined 1d grid if provided, otherwise compute via leja_1d
        :return interp: a TensorProductInterpolator with a refined grid
             or x_new_idx, x_new, interp: where x_new are the newly refined interpolation points (N_new, xdim) and
                                          x_new_idx is the list of indices of these points into interp.xi and interp.yi,
                                          Would use this if you did not provide a callable model to the Interpolator or
                                          you want to manually set yi for each new xi outside this function
        """
        # Initialize a new interpolant with the new refinement levels (child class will provide this method)
        interp = self.new_interpolator(beta)

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
        xi = copy.deepcopy(x_refine) if x_refine is not None else self.leja_1d(num_new_pts, interp.x_bds[dim_refine],
                                                                               z_pts=interp.x_grids[dim_refine],
                                                                               wt_fcn=interp.x_vars[dim_refine].pdf)
        interp.x_grids[dim_refine] = xi

        # Copy yi over at existing interpolation points
        x_new = np.zeros((0, interp.xdim()))
        x_new_idx = []
        tol = 1e-12     # Tolerance for floating point comparison
        j = 0           # Use this idx for iterating over existing yi
        interp.xi = np.zeros((np.prod(new_grid_sizes), self.xdim()))
        interp.yi = np.zeros((np.prod(new_grid_sizes), self.ydim()))
        for i, ele in enumerate(itertools.product(*interp.x_grids)):
            interp.xi[i, :] = ele
            if j < self.xi.shape[0] and np.all(np.abs(self.xi[j, :] - interp.xi[i, :]) < tol):
                # If we already have this interpolation point
                interp.yi[i, :] = self.yi[j, :]
                j += 1
            else:
                # Otherwise, save new interpolation point and its index
                x_new = np.concatenate((x_new, interp.xi[i, :].reshape((1, self.xdim()))))
                x_new_idx.append(i)

        # Evaluate the model at new interpolation points
        interp.wall_time = self.wall_time
        if self._model is None:
            self.logger.warning(f'No model available to evaluate new interpolation points, returning the points '
                                f'to you instead...')
            return x_new_idx, x_new, interp
        elif manual:
            return x_new_idx, x_new, interp
        else:
            y_new = self._model(x_new)  # (N_new, ydim)
            for j in range(y_new.shape[0]):
                interp.yi[x_new_idx[j], :] = y_new[j, :]

            return interp

    @abstractmethod
    def new_interpolator(self, beta):
        """Have the implementing tensor-product class provide a method to instantiate/return a new interpolator object
        :param beta: tuple() specifying new surrogate refinement indices
        """
        pass

    @abstractmethod
    def __call__(self, x):
        """Evaluate the interpolation at points x (child class implements this)
        :param x: (..., xdim) the points to be interpolated, must be within domain of self.xi
        :returns y: (..., ydim) the interpolated value of the qois
        """
        pass

    @staticmethod
    def get_grid_sizes(beta):
        """Compute number of grid points in each dimension"""
        return [2*beta[i] + 1 for i in range(len(beta))]

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
        z_pts = np.atleast_1d(z_pts)

        # Construct leja sequence by maximizing the objective sequentially
        for i in range(N):
            obj_fun = lambda z: -wt_fcn(z) * np.prod(np.abs(z - z_pts))
            res = direct(obj_fun, [z_bds])  # Use global DIRECT optimization over 1d domain
            z_star = res.x
            z_pts = np.concatenate((z_pts, z_star))

        return z_pts


class LagrangeInterpolator(TensorProductInterpolator):
    """Lagrange barycentric interpolator on tensor-product grid"""

    def __init__(self, beta, x_vars, yi=None, model=None, init_grids=True):
        """Construct the interpolator
        :param beta: list(), discretization level in each dimension of xdim
        :param x_vars: list() of BaseRV() objects specifying bounds/pdfs for each input x
        :param yi: the interpolation qoi values, y = (prod(xdims), ydim)
        :param model: Callable as y = model(x), with x = (..., xdim), y = (..., ydim)
        :param init_grids: Whether to compute leja sequences and barycentric weights on init
        """
        self.weights = []   # Barycentric weights for each dimension
        self.j = None       # Cartesian product of all possible refinement indices
        super().__init__(beta, x_vars, yi=yi, model=model, init_grids=init_grids)

        if init_grids:
            # Compute 1d barycentric weights for each grid
            grid_sizes = self.get_grid_sizes(self.beta)
            for n in range(self.xdim()):
                Nx = grid_sizes[n]
                bds = self.x_bds[n]
                grid = self.x_grids[n]
                C = (bds[1] - bds[0]) / 4.0     # Interval capacity (see Berrut and Trefethen 2004)
                xj = grid.reshape((Nx, 1))
                xi = grid.reshape((1, Nx))
                dist = (xj - xi) / C
                np.fill_diagonal(dist, 1)       # Ignore product when i==j
                self.weights.append(1.0 / np.prod(dist, axis=1))  # (Nx,)

            # Construct the multi-indices self.j = [0,...0] --> [Nx, Ny, ...]
            self.j = np.zeros((np.prod(grid_sizes), self.xdim()), dtype=int)
            indices = [np.arange(grid_sizes[n]) for n in range(self.xdim())]
            for i, ele in enumerate(itertools.product(*indices)):
                self.j[i, :] = ele

    def new_interpolator(self, beta):
        """Abstract implementation, admittedly a little messy way to instantiate new interpolator for super()"""
        return LagrangeInterpolator(beta, self.x_vars, model=self._model, init_grids=False)

    def refine(self, beta, manual=False, x_refine=None):
        """Adds required functionality of refining barycentric weights on tensor-product grid
        :param beta: list(), The new refinement level, should only refine one dimension
        :param manual: whether to manually compute model at refinement points
        :param x_refine: (Nx,) use this array as the refined 1d grid if provided, otherwise compute via leja_1d
        :return interp: a LagrangeInterpolator with a refined grid
             or x_new_idx, x_new, interp: where x_new are the newly refined interpolation points (N_new, xdim) and
                                          x_new_idx is the list of indices of these points into interp.xi and interp.yi,
                                          Would use this if you did not provide a callable model to the Interpolator or
                                          you want to manually set yi for each new xi outside this function
        """
        # Refine the tensor-product grid (super handles this)
        if self._model is None or manual:
            x_new_idx, x_new, interp = super().refine(beta, manual=manual, x_refine=x_refine)
            ret = (x_new_idx, x_new, interp)
        else:
            interp = super().refine(beta, manual=manual, x_refine=x_refine)
            ret = interp

        # Find the refinement dimension
        old_grid_sizes = self.get_grid_sizes(self.beta)
        new_grid_sizes = interp.get_grid_sizes(beta)
        dim_refine = 0
        for idx, grid_size in enumerate(new_grid_sizes):
            if grid_size != old_grid_sizes[idx]:
                dim_refine = idx
                break

        # Update barycentric weights in this dimension
        interp.weights = copy.deepcopy(self.weights)
        Nx_old = old_grid_sizes[dim_refine]
        Nx_new = new_grid_sizes[dim_refine]
        old_wts = copy.deepcopy(self.weights[dim_refine])
        new_wts = np.zeros(Nx_new)
        new_wts[:Nx_old] = old_wts
        bds = interp.x_bds[dim_refine]
        C = (bds[1] - bds[0]) / 4.0        # Interval capacity
        xi = interp.x_grids[dim_refine]
        for j in range(Nx_old, Nx_new):
            new_wts[:j] *= (C / (xi[:j] - xi[j]))
            new_wts[j] = np.prod(C / (xi[j] - xi[:j]))
        interp.weights[dim_refine] = new_wts

        # Re-construct the multi-indices interp.j = [0,...0] --> [Nx, Ny, ...]
        interp.j = np.zeros((np.prod(new_grid_sizes), self.xdim()), dtype=int)
        indices = [np.arange(new_grid_sizes[n]) for n in range(self.xdim())]
        for i, ele in enumerate(itertools.product(*indices)):
            interp.j[i, :] = ele

        return ret

    def __call__(self, x):
        """Evaluate the barycentric interpolation at points x (abstract implementation)
        :param x: (..., xdim) the points to be interpolated, must be within domain of self.xi
        :returns y: (..., ydim) the interpolated value of the qois
        """
        # Loop over multi-indices and compute tensor-product lagrange polynomials
        grid_sizes = self.get_grid_sizes(self.beta)
        ydim = self.ydim()
        y = np.zeros(x.shape[:-1] + (ydim,))    # (..., ydim)
        for i in range(np.prod(grid_sizes)):
            L_j = np.zeros(x.shape)             # (..., xdim)

            # Compute univariate Lagrange polynomials in each dimension
            for n in range(self.xdim()):
                j_n = self.j[i, n]
                x_n = x[..., n, np.newaxis]     # (..., 1)

                # Expand axes of interpolation points and weights to match x
                shape = (1,)*len(x_n.shape[:-1]) + (grid_sizes[n],)
                x_j = self.x_grids[n].reshape(shape)  # (1...,Nx)
                w_j = self.weights[n].reshape(shape)  # (1...,Nx)

                # Compute the jth Lagrange basis polynomial L_j(x_n) for this x dimension (in barycentric form)
                c = x_n - x_j
                div_zero_idx = np.isclose(c, 0)     # Track where x is approx equal to an interpolation point x_j
                c[div_zero_idx] = 1                 # Temporarily set to 1 to avoid divide by zero error
                c = w_j / c
                L_j[..., n] = c[..., j_n] / np.sum(c, axis=-1)  # (...) same size as original x
                j_mask = np.full(div_zero_idx.shape, False)
                j_mask[..., j_n] = True
                L_j[np.any(div_zero_idx & j_mask, axis=-1), n] = 1      # Set L_j(x==x_j)=1 for the current j
                L_j[np.any(div_zero_idx & ~j_mask, axis=-1), n] = 0     # Set L_j(x==x_j)=0 for x_j = x_i, i != j

            # Add multivariate basis polynomial contribution to interpolation output
            shape = (1,)*len(x.shape[:-1]) + (ydim,)
            yi = self.yi[i, :].reshape(shape)                   # (1..., ydim)
            L_j = np.prod(L_j, axis=-1)[..., np.newaxis]        # (..., 1)
            y += L_j * yi

        return y
