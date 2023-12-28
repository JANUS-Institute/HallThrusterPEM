"""Script for Markov-Chain Monte Carlo analysis (i.e. Bayesian inference)."""
import numpy as np
import matplotlib.pyplot as plt

from amisc.utils import ndscatter


def dram():
    # TODO: implement DRAM MCMC routine
    pass


if __name__ == '__main__':
    mean = np.array([5, -2, 0.5])
    cov = np.array([[2, 0.4, -0.2], [0.2, 1, -0.2], [-0.2, -0.4, 1]])
    samples = np.random.multivariate_normal(mean, cov.T @ cov, size=10000)
    yt = samples[:, 0] + samples[:, 1]**2 + 2*samples[:, 2]
    ysurr = yt + np.random.randn(*yt.shape)
    err = np.abs(ysurr - yt) / np.abs(yt)
    fig, ax, cb_fig, cb_ax = ndscatter(samples, labels=[r'$\alpha$', r'$\beta$', r'$\gamma$'], plot='scatter',
                                       cmap='plasma', cb_norm='log', z=err)
    plt.show()
