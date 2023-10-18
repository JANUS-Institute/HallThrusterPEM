import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import cm

sys.path.append('..')

from models.simple_models import tanh_func
from surrogates.sparse_grids import TensorProductInterpolator
from utils import UniformRV, print_stats, ax_default


def test_tanh():
    def sampler(N):
        A = np.random.rand(N, 1)*(4 - 1) + 1
        L = np.random.rand(N, 1)*(4 - 1) + 1
        frac = np.random.rand(N, 1)*(8-1) + 1
        return np.concatenate((A, L, frac), axis=-1)

    # Set inputs
    r_pct = 0.999            # Reconstruction accuracy
    interp_dim = 3          # Control interpolation accuracy
    N = 500                 # Size of data matrix
    M = 200                 # Number of FE grid cells

    # Form data matrix
    A = np.zeros((N, M))    # Data matrix
    params = sampler(N)
    zg = np.linspace(0, 4, M)       # b = f(z; x)
    for i in range(N):
        A[i, :] = tanh_func(zg, A=params[i, 0], L=params[i, 1], frac=params[i, 2])

    # Get reduced linear basis
    u, s, vt = np.linalg.svd((A - np.mean(A, axis=0))/np.std(A, axis=0))
    frac = np.cumsum(s**2 / np.sum(s**2))
    idx = int(np.where(frac >= r_pct)[0][0])
    r = idx + 1  # Number of singular values to take
    print(f'r={r} with singular values {s[:r]} and cumsum {frac[:r]}')
    vtr = vt[:r, :]  # (r x M)
    fig, ax = plt.subplots()
    ax.plot(s, '.k')
    ax.set_yscale('log')
    ax_default(ax, 'Index', 'Singular value', legend=False)
    plt.show()

    def model(x):
        bt = tanh_func(zg, A=x[..., 0, np.newaxis], L=x[..., 1, np.newaxis], frac=x[..., 2, np.newaxis])
        bt = ((bt - np.mean(A, axis=0))/np.std(A, axis=0))[..., np.newaxis]
        at = vtr @ bt     # (..., r, 1)
        return {'y': np.squeeze(at, axis=-1)}

    # Interpolate reduced basis
    beta = [interp_dim, interp_dim, interp_dim]
    x_vars = [UniformRV(1, 4), UniformRV(1, 4), UniformRV(1, 8)]  # (A, L, frac)
    interp = TensorProductInterpolator(beta, x_vars, model=model)
    interp.set_yi()

    # Show some random reconstructions
    xtest = np.array([[2.5, 2, 3], [3.5, 1.5, 4], [1.5, 3.5, 2]])
    c = ['r', 'g', 'b']
    ytest = (np.squeeze(vtr.T @ interp(xtest)[..., np.newaxis], axis=-1) * np.std(A, axis=0)) + np.mean(A, axis=0)
    ytruth = tanh_func(zg, A=xtest[..., 0:1], L=xtest[..., 1:2], frac=xtest[..., 2:])
    fig, ax = plt.subplots()
    for i in range(xtest.shape[0]):
        ax.plot(zg, ytruth[i, :], ls='-', c=c[i])
        ax.plot(zg, ytest[i, :], ls='--', c=c[i])
    ax_default(ax, 'z', 'f(z)', legend=False)
    plt.show()

    # Set up test grid for computing errors
    # Ng = 50
    # t_grid = np.linspace(1, 4, Ng)
    # tg1, tg2 = np.meshgrid(t_grid, t_grid)
    # tg1 = tg1.reshape((Ng, Ng, 1))
    # tg2 = tg2.reshape((Ng, Ng, 1))
    # f_cross = [1, 3, 6, 8]
    #
    # # Cross-sections of the f parameter
    # fig, ax = plt.subplots(len(f_cross), r)
    # fig2, ax2 = plt.subplots(len(f_cross), 5)
    # for j, f in enumerate(f_cross):
    #     xt = np.concatenate((tg1, tg2, np.broadcast_to(f, tg1.shape).copy()), axis=-1)
    #     at = model(xt)['y']         # (..., r)
    #     ahat_t = interp(xt)         # (..., r)
    #     bt = tanh_func(zg, A=xt[..., 0, np.newaxis], L=xt[..., 1, np.newaxis], frac=xt[..., 2, np.newaxis])[
    #         ..., np.newaxis]
    #     bhat_t = vtr.T @ at[..., np.newaxis]
    #     bhat_interp = vtr.T @ ahat_t[..., np.newaxis]
    #     ei = ahat_t - at
    #     er = bhat_t - bt
    #     eq = vtr.T @ ei[..., np.newaxis] + er
    #     eq_abs = np.abs(vtr.T @ ei[..., np.newaxis]) + np.abs(er)
    #     eq_also = bhat_interp - bt
    #     assert np.max(np.abs(eq - eq_also)) <= 1e-10
    #     ei_frac = np.squeeze(np.abs(vtr.T @ ei[..., np.newaxis]) / eq_abs, axis=-1)   # (..., M)
    #     er_frac = np.squeeze(np.abs(er) / eq_abs, axis=-1)  # (..., M)
    #     eq_pct = np.squeeze(np.abs(bhat_interp - bt) / bt, axis=-1)
    #
    #     # Plot svd coeffs
    #     for i in range(r):
    #         vmin = min(np.min(at[..., i]), np.min(ahat_t[..., i]))
    #         vmax = max(np.max(at[..., i]), np.max(ahat_t[..., i]))
    #         c1 = ax[j, i].contourf(tg1.squeeze(), tg2.squeeze(), at[..., i], 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    #         plt.colorbar(c1, ax=ax[j, i])
    #
    #         if j == 0:
    #             ax[j, i].set_title(r'{}'.format(f'Coeff $a_{i}$'))
    #         ylabel = r'{}'.format(f'$L$ for $f={f}$') if i == 0 else r'$L$'
    #         ax_default(ax[j, i], r'$A$', ylabel, legend=False)
    #     fig.set_size_inches(r * 3, len(f_cross) * 3)
    #     fig.tight_layout()
    #
    #     # Plot errors (percentiles over FE z grid)
    #     c3 = ax2[j, 0].contourf(tg1.squeeze(), tg2.squeeze(), np.mean(ei_frac, axis=-1), 60, cmap=cm.viridis)
    #     plt.colorbar(c3, ax=ax2[j, 0])
    #     c4 = ax2[j, 1].contourf(tg1.squeeze(), tg2.squeeze(), np.mean(er_frac, axis=-1), 60, cmap=cm.viridis)
    #     plt.colorbar(c4, ax=ax2[j, 1])
    #     c5 = ax2[j, 2].contourf(tg1.squeeze(), tg2.squeeze(), np.percentile(eq_pct, 25, axis=-1), 60, cmap=cm.viridis)
    #     plt.colorbar(c5, ax=ax2[j, 2])
    #     c6 = ax2[j, 3].contourf(tg1.squeeze(), tg2.squeeze(), np.percentile(eq_pct, 50, axis=-1), 60, cmap=cm.viridis)
    #     plt.colorbar(c6, ax=ax2[j, 3])
    #     c7 = ax2[j, 4].contourf(tg1.squeeze(), tg2.squeeze(), np.percentile(eq_pct, 75, axis=-1), 60, cmap=cm.viridis)
    #     plt.colorbar(c7, ax=ax2[j, 4])
    #
    #     if j == 0:
    #         ax2[j, 0].set_title('Mean interp error $|e_i|/|e_q|$')
    #         ax2[j, 1].set_title('Mean reconstr error $|e_r|/|e_q|$')
    #         ax2[j, 2].set_title('25th pct error $|\hat{b} - b|/b$')
    #         ax2[j, 3].set_title('50th pct error $|\hat{b} - b|/b$')
    #         ax2[j, 4].set_title('75th pct error $|\hat{b} - b|/b$')
    #
    #     for i in range(5):
    #         ylabel = r'{}'.format(f'$L$ for $f={f}$') if i == 0 else f'$L$'
    #         ax_default(ax2[j, i], r'$A$', ylabel, legend=False)
    #     fig2.set_size_inches(5*3, len(f_cross)*3)
    #     fig2.tight_layout()
    #
    #     # Print statistics
    #     print(f'-----Results for f={f}-----')
    #     print(f'Mean interpolation error over input space:')
    #     print_stats(np.mean(ei_frac, axis=(0, 1)))
    #     print(f'Mean reconstruction error over input space:')
    #     print_stats(np.mean(er_frac, axis=(0, 1)))
    #     print(f'Min percent error (avg over input space): {np.mean(np.min(eq_pct, axis=-1))}')
    #     print(f'25th pctile percent error (avg over input space): {np.mean(np.percentile(eq_pct, 25, axis=-1))}')
    #     print(f'50th pctile percent error (avg over input space): {np.mean(np.percentile(eq_pct, 50, axis=-1))}')
    #     print(f'Mean percent error (avg over input space): {np.mean(np.mean(eq_pct, axis=-1))}')
    #     print(f'75th pctile percent error (avg over input space): {np.mean(np.percentile(eq_pct, 75, axis=-1))}')
    #     print(f'Max percent error (avg over input space): {np.mean(np.max(eq_pct, axis=-1))}')
    #     print(' ')
    #
    # plt.show()


def test_tanh_2d():
    def sampler(N):
        A = np.random.rand(N, 1) * (4 - 1) + 1
        L = np.random.rand(N, 1) * (4 - 1) + 1
        return np.concatenate((A, L), axis=-1)

    qoi_idx = 6
    m = 1000
    N = 200
    A = np.zeros((m, N))
    params = sampler(m)
    xg = np.linspace(0, 4, N)
    for i in range(m):
        A[i, :] = tanh_func(xg, A=params[i, 0], L=params[i, 1], frac=4)

    u, s, vh = np.linalg.svd(A)  # (mxm), (mxN), (NxN)
    frac = np.cumsum(s / np.sum(s))
    idx = int(np.where(frac >= 0.99)[0][0])
    r = idx + 1  # Number of singular values to take
    print(f'r={r} with singular values {s[:r]} and cumsum {frac[:r]}')
    uk = u[:, :r]
    sk = np.diag(s[:r])
    vtk = vh[:r, :]  # (r x N)
    A_approx = uk @ sk @ vtk
    print(f'||A-A_tilde||_max = {np.max(np.abs(A - A_approx))}')

    # Interpolate reduced basis
    beta = [2, 2]
    x_vars = [UniformRV(1, 4), UniformRV(1, 4)]  # (A, L)

    def model(x):
        y = tanh_func(xg, A=x[..., 0, np.newaxis], L=x[..., 1, np.newaxis], frac=4)[..., np.newaxis]
        yr = vtk @ y  # (..., r, 1)
        return {'y': np.squeeze(yr, axis=-1)}

    interp = TensorProductInterpolator(beta, x_vars, model=model)
    interp.set_yi()
    Ntheta = 50
    theta_grid = np.linspace(1, 4, Ntheta)
    tg1, tg2 = np.meshgrid(theta_grid, theta_grid)
    tg1 = tg1.reshape((Ntheta, Ntheta, 1))
    tg2 = tg2.reshape((Ntheta, Ntheta, 1))
    x = np.concatenate((tg1, tg2), axis=-1)
    z = model(x)[..., qoi_idx, np.newaxis]
    z_interp = interp(x)[..., qoi_idx, np.newaxis]
    error = np.abs(z_interp - z)

    # Refine interpolant
    beta2 = [3, 3]
    interp2 = TensorProductInterpolator(beta2, x_vars, model=model)
    interp2.set_yi()
    z2_interp = interp2(x)[..., qoi_idx, np.newaxis]
    error2 = np.abs(z2_interp - z)
    vmin = min(np.min(z_interp), np.min(z), np.min(z2_interp))
    vmax = max(np.max(z_interp), np.max(z), np.max(z2_interp))
    emin = min(np.min(error), np.min(error2))
    emax = max(np.max(error), np.max(error2))

    fig, ax = plt.subplots(2, 3)
    c1 = ax[0, 0].contourf(tg1.squeeze(), tg2.squeeze(), z.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=ax[0, 0])
    ax[0, 0].set_title('True function')
    ax_default(ax[0, 0], r'$A$', r'$L$', legend=False)
    c2 = ax[0, 1].contourf(tg1.squeeze(), tg2.squeeze(), z_interp.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    ax[0, 1].plot(interp.xi[:, 0], interp.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c2, ax=ax[0, 1])
    ax[0, 1].set_title('Interpolant')
    ax_default(ax[0, 1], r'$A$', '', legend=False)
    c3 = ax[0, 2].contourf(tg1.squeeze(), tg2.squeeze(), error.squeeze(), 60, cmap=cm.viridis, vmin=emin, vmax=emax)
    ax[0, 2].plot(interp.xi[:, 0], interp.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c3, ax=ax[0, 2])
    ax[0, 2].set_title('Absolute error')
    ax_default(ax[0, 2], r'$A$', '', legend=False)
    c1 = ax[1, 0].contourf(tg1.squeeze(), tg2.squeeze(), z.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=ax[1, 0])
    ax[1, 0].set_title('True function')
    ax_default(ax[1, 0], r'$A$', r'$L$', legend=False)
    c2 = ax[1, 1].contourf(tg1.squeeze(), tg2.squeeze(), z2_interp.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    ax[1, 1].plot(interp2.xi[:, 0], interp2.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c2, ax=ax[1, 1])
    ax[1, 1].set_title('Refined')
    ax_default(ax[1, 1], r'$A$', '', legend=False)
    c3 = ax[1, 2].contourf(tg1.squeeze(), tg2.squeeze(), error2.squeeze(), 60, cmap=cm.viridis, vmin=emin, vmax=emax)
    ax[1, 2].plot(interp2.xi[:, 0], interp2.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c3, ax=ax[1, 2])
    ax[1, 2].set_title('Absolute error')
    ax_default(ax[1, 2], r'$A$', '', legend=False)
    fig.set_size_inches(15, 11)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_tanh()
    # test_tanh_2d()
