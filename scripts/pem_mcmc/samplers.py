import math
import random
from os import PathLike
from typing import Any

import amisc
import amisc.distribution as distributions
import numpy as np
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis

import pem_mcmc.io as io
from pem_mcmc import log_posterior


class Sampler:
    variables: list[amisc.Variable]
    base_vars: dict[amisc.Variable, Any]
    init_sample_file: PathLike | None
    init_cov_file: PathLike | None
    system: amisc.System

    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        self.variables = variables
        self.data = data
        self.system = system
        self.base_vars = base_vars
        self.opts = opts
        self.init_sample_file = init_sample_file
        self.init_cov_file = init_cov_file
        self.logpdf = lambda x: log_posterior(dict(zip(variables, x)), data, system, base_vars, opts)
        self._init_sample = None
        self._init_cov = None

    def cov(self):
        return self.initial_cov()

    def initial_sample(self):
        if self._init_sample is None:
            # Read initial sample from file or create it from the base variables dict
            if self.init_sample_file is None:
                self._init_sample = np.array([self.base_vars[p] for p in self.variables])
            else:
                index_map = {p.name: i for (i, p) in enumerate(self.variables)}
                self._init_sample = np.zeros(len(self.variables))
                var_dict = io.read_dlm(self.init_sample_file)
                for k, v in var_dict.items():
                    i = index_map[k]
                    self._init_sample[i] = v[0]

        return self._init_sample

    def initial_cov(self):
        if self._init_cov is None:
            index_map = {p.name: i for (i, p) in enumerate(self.variables)}
            if self.init_cov_file is None:
                variances = np.ones(len(self.variables))

                # Use variable distributions to estimate covariance
                for i, p in enumerate(self.variables):
                    dist = self.system.inputs()[p].distribution
                    if isinstance(dist, distributions.Uniform) or isinstance(dist, distributions.LogUniform):
                        lb, ub = dist.dist_args
                        std = (ub - lb) / 4
                    elif isinstance(dist, distributions.Normal):
                        std = dist.dist_args[1]
                    elif isinstance(dist, distributions.LogNormal):
                        std = dist.base ** dist.dist_args[1]
                    else:
                        raise ValueError(
                            f"Unsupported distribution {dist}. Currently only `Uniform`, `LogUniform`, `Normal` and `LogNormal` are supported."  # noqa: E501
                        )

                    variances[i] = self.system.inputs()[p].normalize(std) ** 2
                self._init_cov = np.diag(variances)
            else:
                # Construct covariance from file
                # We support the variables being in a different order so we build the covariance up from the index map.
                cov_dict = io.read_dlm(self.init_cov_file)
                N = len(self.variables)
                self._init_cov = np.zeros((N, N))
                for i, (key1, column) in enumerate(cov_dict.items()):
                    i1 = index_map[key1]
                    for j, (var, key2) in enumerate(zip(column, cov_dict.keys())):
                        i2 = index_map[key2]
                        self._init_cov[i1, i2] = var

            # Verify that the covariance matrix is positive-definite before proceeding.
            # This throws an exception if not.
            np.linalg.cholesky(self._init_cov)
        return self._init_cov


class PriorSampler(Sampler):
    """
    Samples from prior distribution, run model, and evaluates posterior probability
    """

    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        super().__init__(variables, data, system, base_vars, opts, init_sample_file, init_cov_file)

    def __iter__(self):
        return self

    def __next__(self):
        sample = np.array([var.normalize(var.distribution.sample((1,)))[0] for var in self.variables])
        logp = self.logpdf(sample)
        return sample, logp, np.isfinite(logp)


class PreviousRunSampler(Sampler):
    """
    Samples (with replacement) from a previous MCMC run.
    """

    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        prev_run_file: PathLike,
        burn_fraction: float = 0.5,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        super().__init__(variables, data, system, base_vars, opts, init_sample_file, init_cov_file)
        _vars, _samples, _, accepted, _ = io.read_output_file(prev_run_file)

        # Figure out the first index we should sample.
        self.start_index = math.floor(burn_fraction * len(_samples))

        # associate variable names with columns and remove burned samples
        _samples = np.array(_samples)
        col_dict = {system.inputs()[v]: _samples[self.start_index :, i] for (i, v) in enumerate(_vars)}

        # reorder to match input varible list.
        # this will error if variable lists don't match
        self.samples = np.array([col_dict[v] for v in self.variables]).T
        self.accepted = accepted[self.start_index :]

    def __iter__(self):
        return self

    def sample_index(self):
        return random.randint(0, self.samples.shape[0] - 1)

    def __next__(self):
        # draw until we get an accepted sample
        index = self.sample_index()

        while not self.accepted[index]:
            index = self.sample_index()

        sample = self.samples[index, :]
        logp = self.logpdf(sample)
        return sample, logp, np.isfinite(logp)


class DRAMSampler(Sampler):
    """
    Samples using delayed rejection adaptive metropolis
    """

    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        super().__init__(variables, data, system, base_vars, opts, init_sample_file, init_cov_file)

        self.sampler = DelayedRejectionAdaptiveMetropolis(
            self.logpdf,
            self.initial_sample(),
            self.initial_cov(),
            adapt_start=10,
            eps=1e-6,
            sd=None,
            interval=1,
            level_scale=1e-1,
        )

    def cov(self):
        return self.sampler.cov_chol

    def __iter__(self):
        return iter(self.sampler)
