"""Script for doing Sobol' sensitivity analysis."""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from uqtils import ax_default

from hallmd.models.examples import ishigami


def sobol_sa(model, sampler, num_samples, qoi_idx=None, qoi_labels=None, param_labels=None, plot=True, verbose=True,
             cmap='viridis'):
    """Perform a global Sobol sensitivity analysis.

    :param model: callable as `y=model(x)`, with `y=(...,ydim)`, `x=(...,xdim)`
    :param sampler: callable as `x=sampler(shape)`, with `x=(*shape, xdim)`
    :param num_samples: number of samples
    :param qoi_idx: list of indices of model output to report results for
    :param qoi_labels: list of labels for plotting QoIs
    :param param_labels: list of labels for plotting input parameters
    :param plot: whether to plot bar/pie charts
    :param verbose: whether to print S1/ST/S2 results to the console
    :param cmap: str() specifier of plt.colormap for bar/pie charts
    :return: `S1`, `S2`, `ST`, the first, second, and total order Sobol' indices
    """
    # Get sample matrices (N, xdim)
    A = sampler((num_samples,))
    B = sampler((num_samples,))
    xdim = A.shape[-1]
    AB = np.tile(np.expand_dims(A, axis=-2), (1, xdim, 1))
    BA = np.tile(np.expand_dims(B, axis=-2), (1, xdim, 1))
    for i in range(xdim):
        AB[:, i, i] = B[:, i]
        BA[:, i, i] = A[:, i]

    # Evaluate the model; (xdim+2)*N evaluations required
    fA = model(A)       # (N, ydim)
    fB = model(B)       # (N, ydim)
    fAB = model(AB)     # (N, xdim, ydim)
    fBA = model(BA)     # (N, xdim, ydim)
    ydim = fA.shape[-1]

    # Normalize model outputs to N(0, 1) for better stability
    Y = np.concatenate((fA, fB, fAB.reshape((-1, ydim)), fBA.reshape((-1, ydim))), axis=0)
    mu, std = np.mean(Y, axis=0), np.std(Y, axis=0)
    fA = (fA - mu) / std
    fB = (fB - mu) / std
    fAB = (fAB - mu) / std
    fBA = (fBA - mu) / std

    # Compute sensitivity indices
    vY = np.var(np.concatenate((fA, fB), axis=0), axis=0)   # (ydim,)
    fA = np.expand_dims(fA, axis=-2)                        # (N, 1, ydim)
    fB = np.expand_dims(fB, axis=-2)                        # (N, 1, ydim)
    S1 = fB * (fAB - fA) / vY                               # (N, xdim, ydim)
    ST = 0.5 * (fA - fAB)**2 / vY                           # (N, xdim, ydim)

    # Second-order indices
    Vij = np.expand_dims(fBA, axis=2) * np.expand_dims(fAB, axis=1) - \
          np.expand_dims(fA, axis=1) * np.expand_dims(fB, axis=1)   # (N, xdim, xdim, ydim)
    si = fB * (fAB - fA)
    Vi = np.expand_dims(si, axis=2)
    Vj = np.expand_dims(si, axis=1)
    S2 = (Vij - Vi - Vj) / vY                               # (N, xdim, xdim, ydim)

    # Get mean values and MC error
    S1_est = np.mean(S1, axis=0)
    S1_se = np.sqrt(np.var(S1, axis=0) / num_samples)
    S2_est = np.mean(S2, axis=0)
    S2_se = np.sqrt(np.var(S2, axis=0) / num_samples)
    ST_est = np.mean(ST, axis=0)
    ST_se = np.sqrt(np.var(ST, axis=0) / num_samples)

    # Set defaults for qoi indices/labels
    if qoi_idx is None:
        qoi_idx = list(np.arange(ydim))
    if qoi_labels is None:
        qoi_labels = [f'QoI {i}' for i in range(len(qoi_idx))]
    if param_labels is None:
        param_labels = [f'x{i}' for i in range(xdim)]

    # Print results
    if verbose:
        print(f'{"QoI":>10} {"Param":>10} {"S1_mean":>10} {"S1_err":>10} {"ST_mean":>10} {"ST_err":>10}')
        for i in range(len(qoi_idx)):
            for j in range(xdim):
                q = qoi_idx[i]
                print(f'{qoi_labels[i]:>10} {param_labels[j]:>10} {S1_est[j, q]: 10.3f} {S1_se[j, q]: 10.3f} '
                      f'{ST_est[j, q]: 10.3f} {ST_se[j, q]: 10.3f}')

        print(f'\n{"QoI":>10} {"2nd-order":>20} {"S2_mean":>10} {"S2_err":>10}')
        for i in range(len(qoi_idx)):
            for j in range(xdim):
                for k in range(j+1, xdim):
                    q = qoi_idx[i]
                    print(f'{qoi_labels[i]:>10} {"("+param_labels[j]+", "+param_labels[k]+")":>20} '
                          f'{S2_est[j, k, q]: 10.3f} {S2_se[j, k, q]: 10.3f}')

        S1_total = np.sum(S1_est, axis=0)       # (ydim,)
        S2_total = np.zeros((ydim,))            # (ydim,)
        for i in range(xdim):
            for j in range(i+1, xdim):
                S2_total += S2_est[i, j, :]     # sum the upper diagonal
        print(f'\n{"QoI":>10} {"S1 total":>10} {"S2 total":>10} {"Higher order":>15}')
        for i in range(len(qoi_idx)):
            q = qoi_idx[i]
            print(f'{qoi_labels[i]:>10} {S1_total[q]: 10.3f} {S2_total[q]: 10.3f} {1 - S1_total[q] - S2_total[q]: 15.3f}')

    if plot:
        # Plot bar chart of S1, ST
        c = plt.get_cmap(cmap)
        fig, axs = plt.subplots(1, len(qoi_idx))
        for i in range(len(qoi_idx)):
            ax = axs[i] if len(qoi_idx) > 1 else axs
            q = qoi_idx[i]
            z = st.norm.ppf(1 - (1-0.95)/2)  # get z-score from N(0,1), assuming CLT at n>30
            x = np.arange(xdim)
            width = 0.2
            ax.bar(x - width / 2, S1_est[:, q], width, color=c(0.1), yerr=S1_se[:, q] * z,
                   label=r'$S_1$', capsize=3, linewidth=1, edgecolor=[0, 0, 0])
            ax.bar(x + width / 2, ST_est[:, q], width, color=c(0.9), yerr=ST_se[:, q] * z,
                   label=r'$S_{T}$', capsize=3, linewidth=1, edgecolor=[0, 0, 0])
            ax_default(ax, "Model parameters", "Sobol' index", legend=True)
            ax.set_xticks(x, param_labels)
            ax.set_ylim(bottom=0)
            ax.set_title(qoi_labels[i])
        fig.set_size_inches(4*len(qoi_idx), 4)
        fig.tight_layout()
        plt.show()

        # Plot pie chart of S1, S2, higher-order
        fig, axs = plt.subplots(1, len(qoi_idx))
        for i in range(len(qoi_idx)):
            ax = axs[i] if len(qoi_idx) > 1 else axs
            q = qoi_idx[i]
            values = []
            labels = []
            s12_other = 0
            thresh = 0.05    # Only show indices with > 5% effect
            for j in range(xdim):
                if S1_est[j, q] > thresh:
                    values.append(S1_est[j, q])
                    labels.append(param_labels[j])
                else:
                    s12_other += max(S1_est[j, q], 0)
            for j in range(xdim):
                for k in range(j+1, xdim):
                    if S2_est[j, k, q] > thresh:
                        values.append(S2_est[j, k, q])
                        labels.append("("+param_labels[j]+", "+param_labels[k]+")")
                    else:
                        s12_other += max(S2_est[j, k, q], 0)

            values.append(max(s12_other, 0))
            labels.append(r'Other $S_1$, $S_2$')
            s_higher = max(1 - np.sum(values), 0)
            values.append(s_higher)
            labels.append(r'Higher order')

            # Adjust labels to show percents, sort by value, and threshold small values for plotting
            labels = [f"{label}, {100*values[i]:.1f}%" if values[i] > thresh else
                      f"{label}, <{max(0.5, round(100*values[i]))}%" for i, label in enumerate(labels)]
            values = [val if val > thresh else max(0.02, val) for val in values]
            labels, values = list(zip(*sorted(zip(labels, values), reverse=True, key=lambda ele: ele[1])))

            # Generate pie chart
            colors = c(np.linspace(0, 1, len(values)-2))
            gray_idx = [idx for idx, label in enumerate(labels) if label.startswith('Higher') or label.startswith('Other')]
            pie_colors = np.empty((len(values), 4))
            c_idx = 0
            for idx in range(len(values)):
                if idx in gray_idx:
                    pie_colors[idx, :] = [0.7, 0.7, 0.7, 1]
                else:
                    pie_colors[idx, :] = colors[c_idx, :]
                    c_idx += 1
            radius = 2
            wedges, label_boxes = ax.pie(values, colors=pie_colors, radius=radius, startangle=270,
                                         shadow=True, counterclock=False, frame=True,
                                         wedgeprops=dict(linewidth=1.5, width=0.6*radius, edgecolor='w'),
                                         textprops={'color': [0, 0, 0, 1], 'fontsize': 10, 'family': 'serif'})
            kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center", fontsize=9, family='serif',
                      bbox=dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0))

            # Put annotations with arrows to each wedge (coordinate system is relative to center of pie)
            for j, wed in enumerate(wedges):
                ang = (wed.theta2 - wed.theta1) / 2. + wed.theta1
                x = radius * np.cos(np.deg2rad(ang))
                y = radius * np.sin(np.deg2rad(ang))
                ax.scatter(x, y, s=10, c='k')
                kw["horizontalalignment"] = "right" if int(np.sign(x)) == -1 else "left"
                kw["arrowprops"].update({"connectionstyle": f"angle,angleA=0,angleB={ang}"})
                y_offset = 0.2 if j == len(labels) - 1 else 0
                ax.annotate(labels[j], xy=(x, y), xytext=((radius+0.2)*np.sign(x), 1.3*y - y_offset), **kw)
            ax.set(aspect="equal")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.get_yaxis().set_ticks([])
            ax.get_xaxis().set_ticks([])
            ax.set_title(qoi_labels[i])
        fig.set_size_inches(3*radius*len(qoi_idx), 2.5*radius)
        fig.tight_layout()
        fig.subplots_adjust(left=0.15, right=0.75)
        plt.show()

    return S1, S2, ST


if __name__ == '__main__':
    model = lambda x: ishigami(x)['y']
    def lhc_sampler(shape):
        N, dim = np.prod(shape), 3
        x = st.qmc.LatinHypercube(d=dim).random(n=N)
        x = st.qmc.scale(x, [-np.pi]*dim, [np.pi]*dim).reshape(shape + (dim,))
        return x
    rand_sampler = lambda shape: np.random.rand(*shape, 3)*(2*np.pi) - np.pi
    sobol_sa(model, lhc_sampler, 10000, cmap='summer')
