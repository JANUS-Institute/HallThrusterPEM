"""Module for inverse uncertainty quantification tasks (i.e. MCMC)"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, AutoLocator, FuncFormatter
import matplotlib
import scipy.stats as st


def ndscatter(samples, labels=None, tick_fmts=None, plot='scatter', cmap='viridis', bins=20, z=None, cb_label=None,
              cb_norm='linear', subplot_size=3):
    """Triangle scatter plots of n-dimensional samples
    :param samples: (N, dim) samples to plot
    :param labels: list() of length dim specifying axis labels
    :param tick_fmts: list() of str() specifying str.format() for ticks, e.g ['{x: >10.2f}', ...], of length dim
    :param plot: str(), 'hist' for 2d hist plot, 'kde' for kernel density estimation, or 'scatter' (default)
    :param cmap: str(), the matplotlib string specifier of a colormap
    :param bins: int, number of bins in each dimension for histogram marginals
    :param z: (N,) a performance metric corresponding to samples, used to color code the scatter plot if provided
    :param cb_label: str(), label for colorbar (if z is provided)
    :param cb_norm: str() or plt.colors.Normalize, normalization method for plotting z on scatter plot
    :param subplot_size: size in inches of a single 2d marginal subplot
    """
    N, dim = samples.shape
    x_min = np.min(samples, axis=0)
    x_max = np.max(samples, axis=0)
    if labels is None:
        labels = [f"x{i}" for i in range(dim)]
    if z is None:
        z = np.zeros(N)
    if cb_label is None:
        cb_label = 'Performance metric'

    def tick_format_func(value, pos):
        if value > 1:
            return f'{value:.2f}'
        if value > 0.01:
            return f'{value:.4f}'
        if value < 0.01:
            return f'{value:.2E}'
    default_ticks = FuncFormatter(tick_format_func)
    # if tick_fmts is None:
    #     tick_fmts = ['{x:.2G}' for i in range(dim)]

    # Set up triangle plot formatting
    fig, axs = plt.subplots(dim, dim, sharex='col', sharey='row')
    for i in range(dim):
        for j in range(dim):
            ax = axs[i, j]
            if i == j:                      # 1d marginals on diagonal
                ax.get_shared_y_axes().remove(ax)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                if i == 0:
                    ax.get_yaxis().set_ticks([])
            if j > i:                       # Clear the upper triangle
                ax.axis('off')
            if i == dim - 1:                # Bottom row
                ax.set_xlabel(labels[j])
                ax.xaxis.set_major_locator(AutoLocator())
                formatter = StrMethodFormatter(tick_fmts[j]) if tick_fmts is not None else default_ticks
                ax.xaxis.set_major_formatter(formatter)
            if j == 0 and i > 0:            # Left column
                ax.set_ylabel(labels[i])
                ax.yaxis.set_major_locator(AutoLocator())
                formatter = StrMethodFormatter(tick_fmts[i]) if tick_fmts is not None else default_ticks
                ax.yaxis.set_major_formatter(formatter)

    # Plot marginals
    for i in range(dim):
        for j in range(dim):
            ax = axs[i, j]
            if i == j:                      # 1d marginals (on diagonal)
                c = plt.get_cmap(cmap)(0)
                if plot == 'kde':
                    kernel = st.gaussian_kde(samples[:, i])
                    x = np.linspace(x_min[i], x_max[i], 1000)
                    ax.fill_between(x, y1=kernel(x), y2=0, lw=0, alpha=0.3, facecolor=c)
                    ax.plot(x, kernel(x), ls='-', c=c, lw=1.5)
                else:
                    ax.hist(samples[:, i], edgecolor='black', color=c, density=True, alpha=0.5,
                            linewidth=1.2, bins='auto')
            if j < i:                       # 2d marginals (lower triangle)
                ax.set_xlim([x_min[j], x_max[j]])
                ax.set_ylim([x_min[i], x_max[i]])
                if plot == 'scatter':
                    sc = ax.scatter(samples[:, j], samples[:, i], s=1.5, c=z, cmap=cmap, norm=cb_norm)
                elif plot == 'hist':
                    ax.hist2d(samples[:, j], samples[:, i], bins=bins, density=True, cmap=cmap)
                elif plot == 'kde':
                    kernel = st.gaussian_kde(samples[:, [j, i]].T)
                    xg, yg = np.meshgrid(np.linspace(x_min[j], x_max[j], 60), np.linspace(x_min[i], x_max[i], 60))
                    x = np.vstack([xg.ravel(), yg.ravel()])
                    zg = np.reshape(kernel(x), xg.shape)
                    ax.contourf(xg, yg, zg, 5, cmap=cmap, alpha=0.9)
                    ax.contour(xg, yg, zg, 5, colors='k', linewidths=1.5)
                else:
                    raise NotImplementedError('This plot type is not known. plot=["hist", "kde", "scatter"]')

    fig.set_size_inches(subplot_size * dim, subplot_size * dim)
    fig.tight_layout()
    plt.show()

    # Plot colorbar in standalone figure
    if np.max(z) > 0 and plot == 'scatter':
        cb_fig, cb_ax = plt.subplots(figsize=(1.5, 6))
        cb_fig.subplots_adjust(right=0.7)
        cb_fig.colorbar(sc, cax=cb_ax, orientation='vertical', label=cb_label)
        cb_fig.tight_layout()
        plt.show()
        return fig, axs, cb_fig, cb_ax

    return fig, axs


if __name__ == '__main__':
    mean = np.array([5, -2, 0.5])
    cov = np.array([[2, 0.4, -0.2], [0.2, 1, -0.2], [-0.2, -0.4, 1]])
    samples = np.random.multivariate_normal(mean, cov.T @ cov, size=10000)
    yt = samples[:, 0] + samples[:, 1]**2 + 2*samples[:, 2]
    ysurr = yt + np.random.randn(*yt.shape)
    err = np.abs(ysurr - yt) / np.abs(yt)
    print(cov.T @ cov)
    ndscatter(samples, labels=[r'$\alpha$', r'$\beta$', r'$\gamma$'], plot='scatter', cmap='plasma',
              cb_norm='log', z=err)
