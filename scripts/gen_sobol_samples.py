import argparse
import math
import os
from pathlib import Path

import amisc.distribution
import numpy as np
import pem_mcmc as mcmc
import pem_mcmc.io
import scipy.stats

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("-n", type=int)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--distribution", type=str, default="prior")
parser.add_argument("--posterior-dir", type=str)
parser.add_argument("--qmc", action="store_true")


class RdSampler:
    """R-d quasi-random sampler, from https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences"""

    def __init__(self, d, skip=0, seed=None):
        # validate seed
        if seed is None:
            self.seed = np.array([0] * d)
        elif type(seed) is list:
            if len(seed) == d:
                self.seed = np.array(seed)
            else:
                raise ValueError("Length of seed list must be equal to sample dimension")
        else:
            self.seed = np.array([seed] * d)

        # compute phi_d, the root of x^d - x - 1 = 0
        self.d = d
        self.phi = RdSampler.compute_root(d)

        # generate alpha_i
        self.alpha = np.array([1 / (self.phi) ** (j + 1) for j in range(d)])

        # initialize sampler, skipping to the k-th sample if requested
        self.current = self.seed
        self.index = 0
        if skip > 0:
            self.sample(skip)

    @staticmethod
    def compute_root(d):
        """
        Parameters
        ----------
        d : Int
            dimensionality of polynomial

        Returns
        -------
        phi: float
            positive, real root of x^d - x - 1 = 0
        """
        # compute phi using recurrance relation
        # phi = (1 + (1 + (1 + (1 + ...)^p )^p )^p )^p
        # where p = 1 / (d+1)
        p = 1 / (d + 1)
        phi = 1.5
        tol = 1e-16  # should be enough within fp64 accuracy
        max_iter = 50
        res = phi
        for i in range(max_iter):
            phi_new = (1 + phi) ** p
            res = np.abs(phi - phi_new)
            phi = phi_new
            if res < tol:
                break

        return phi

    def sample(self, num_samples, lb=None, ub=None):
        # initialize output array
        samples = np.empty((num_samples + 1, self.d))
        samples[0] = self.current

        for i in range(1, num_samples + 1):
            samples[i] = samples[i - 1] + self.alpha

        # restrict to range 0-1
        samples = samples % 1

        # scale to proper range
        if lb is not None or ub is not None:
            if lb is None:
                lb = np.zeros(self.d)
            if ub is None:
                ub = np.ones(self.d)
            boxsize = ub - lb
            samples = boxsize * samples + lb

        # update current samples
        self.current = samples[-1, :]
        self.index += num_samples
        return samples[1:]


def transform_uniform(x, dist):
    lb, ub = dist.dist_args
    scale = ub - lb
    x1 = lb + scale * x
    return x1


def transform_loguniform(x, dist):
    lb, ub = dist.dist_args
    c = 1 / np.log(dist.base)
    return dist.base ** (x * c * (np.log(ub) - np.log(lb)) + c * np.log(lb))


def transform_normal(x, dist):
    mu, std = dist.dist_args
    return scipy.stats.norm.ppf(x, mu, std)


def transform_lognormal(x, dist):
    mu, std = dist.args
    normal_samples = scipy.stats.norm.ppf(x, mu, std)
    return dist.base**normal_samples


def sample_variables(N, variables, system, qmc=False):
    """
    Generate N quasirandom samples according to the 1-D marginal distributions.
    Returns a matrix of dimension (N, d), where d is the number of distributions provided.
    The samples are generated according to R^n quasi-monte-carlo sequences and so admit nice subsampling and restart properties.
    See this article for details on the sequence: https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """  # noqa: E501
    d = len(variables)

    if qmc:
        # Generate uniform samples on the interval 0 - 1
        sampler = RdSampler(d)
        samples = sampler.sample(N)
        # Transform samples to target distributions
        for id, variable in enumerate(variables):
            dist = variable.distribution
            if isinstance(dist, amisc.distribution.Uniform):
                samples_unnormalized = transform_uniform(samples[:, id], dist)
                samples[:, id] = variable.normalize(samples_unnormalized)
            elif isinstance(dist, amisc.distribution.LogUniform):
                samples_unnormalized = transform_loguniform(samples[:, id], dist)
                samples[:, id] = variable.normalize(samples_unnormalized)
            elif isinstance(dist, amisc.distribution.Normal):
                samples_unnormalized = transform_normal(samples[:, id], dist)
                samples[:, id] = variable.normalize(samples_unnormalized)
            elif isinstance(dist, amisc.distribution.LogNormal):
                samples_unnormalized = transform_normal(samples[:, id], dist)
                samples[:, id] = variable.normalize(samples_unnormalized)
            else:
                raise ValueError(
                    f"Unsupported distribution {dist}. Currently only `Uniform`, `LogUniform`, `Normal` and `LogNormal` are supported."  # noqa: E501
                )
    else:
        var_names = [var.name for var in variables]
        sample_dict = system.sample_inputs(size=N, normalize=True, use_pdf=True, include=var_names)
        samples = np.zeros((N, d))
        for id, variable in enumerate(variables):
            samples[:, id] = sample_dict[variable.name]

    assert samples.shape == (N, d)
    return samples


def generate_sampling_plan(N, variables, system, distribution="prior", posterior_dir=".", qmc=False):
    """
    Generate N samples of variables given a list of `d` amisc variables.
    These are then laid out into three sample tensors (A, B, C) following Saltelli (2002) and the nomenclature of Issan et al (2023).
    The dimensions of these tensors are A: (N, d), B: (N, d), C: (d, N, d).
    """  # noqa: E501
    d = len(variables)

    if distribution == "prior":
        # Sample prior distribution
        AB = sample_variables(2 * N, variables, system, qmc=qmc)
    else:
        # Sample posterior distribution
        assert posterior_dir is not None
        logfile = Path(posterior_dir) / "mcmc" / "mcmc.csv"
        read_variables, samples, _, _, _ = pem_mcmc.io.read_output_file(logfile)
        burn_fraction = 0.5
        samples = samples[math.floor(burn_fraction * samples.shape[0]) :, :]

        rand_indices = np.random.choice(np.arange(samples.shape[0]), 2 * N, replace=False)

        sample_dict = {}
        for id, varname in enumerate(read_variables):
            sample_dict[varname] = samples[rand_indices, id]

        AB = np.zeros((2 * N, len(variables)))
        for id, varname in enumerate(variables):
            AB[:, id] = sample_dict[varname]

    A = AB[:N, :]
    B = AB[N:, :]
    C = np.repeat(B[None, :, :], d, axis=0)

    # Copy ith column of A over to C
    for i in range(d):
        C[i, :, i] = A[:, i]

    C = np.array(C)
    assert A.shape == (N, d)
    assert B.shape == (N, d)
    assert C.shape == (d, N, d)

    return A, B, C


def create_output_file(filename, samples, header, delimiter=","):
    (N, _) = samples.shape
    with open(filename, "w") as fd:
        print(header, file=fd)
        for i in range(N):
            mcmc.write_sample_row_fd(fd, i, samples[i, :], 0.0, True, delimiter)


if __name__ == "__main__":
    args = parser.parse_args()

    system = mcmc.load_system(args.config)
    variables = mcmc.load_calibration_variables(system, sort='name')

    A, B, C = generate_sampling_plan(
        args.n, variables, system, distribution=args.distribution, posterior_dir=args.posterior_dir, qmc=args.qmc
    )

    # Save samples to output files
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    header = ",".join(["id"] + [p.name for p in variables] + ["log_posterior", "accepted"])

    create_output_file(output_dir / "A.csv", A, header)
    create_output_file(output_dir / "B.csv", B, header)

    for i in range(len(variables)):
        create_output_file(output_dir / f"C_{i:02d}.csv", C[i, ...], header)

    print(A.shape)
