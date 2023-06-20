import numpy as np
import itertools


def level_to_grid(beta, k=2):
    return tuple([k*i + 1 for i in beta])


def get_grid(beta):
    N = level_to_grid(beta)
    xp = np.zeros((np.prod(N), len(beta)), dtype=np.uint8)
    x_grids = [np.arange(N[d]) for d in range(len(beta))]
    for i, ele in enumerate(itertools.product(*x_grids)):
        xp[i, :] = ele

    return xp


def get_global_indices(xj, xM):
    # xj must be a subset of xM
    xj = np.expand_dims(xj, axis=1)                 # (Lj, 1, d)
    xM = np.expand_dims(xM, axis=0)                 # (1, LM, d)
    diff = xM - xj                                  # (Lj, LM, d)
    idx = np.count_nonzero(diff, axis=-1) == 0      # (Lj, LM)
    idx = np.nonzero(idx)[1]                        # (Lj,)

    return idx


def get_beta_info(beta_j, x_M):
    # Sub-grid
    N_j = level_to_grid(beta_j)
    L_j = np.prod(N_j)
    x_j = get_grid(beta_j)
    i_j = get_global_indices(x_j, x_M)
    diff = i_j[1:] - i_j[0:-1]
    G_j = np.nonzero(diff > 1)[0]  # First index with diff > 1 marks the end of the group
    G_j = G_j[0] + 1 if G_j.shape[0] > 0 else L_j
    g_j = int(L_j / G_j)
    gj_groups = [x_j[i:i + G_j, :] for i in range(0, L_j, G_j)]

    # Get neighbors
    Q_j = []
    for i in range(len(beta_j)):
        if beta_j[i]:
            beta_q = np.array(beta_j)
            beta_q[i] -= 1
            beta_q = tuple(beta_q)
            Q_j.append(beta_q)

    # Get index of first unique group in each neighbor
    gamma_j = {}
    for beta_q in Q_j:
        x_q = get_grid(beta_q)
        i_q = get_global_indices(x_q, x_M)
        L_q = np.prod(level_to_grid(beta_q))
        diff = i_q[1:] - i_q[0:-1]
        G_q = np.nonzero(diff > 1)[0]
        G_q = G_q[0] + 1 if G_q.shape[0] > 0 else L_q
        g_q = int(L_q / G_q)
        iq_groups = [list(i_q[i:i + G_q]) for i in range(0, L_q, G_q)]
        gamma_j[str(beta_q)] = iq_groups

    return N_j, L_j, G_j, g_j, gj_groups, gamma_j


def group_size(beta_j, beta_M):
    N_j = level_to_grid(beta_j)
    dim = len(beta_j)

    def chi(d):
        if d == 0:
            return 1
        if beta_j[d] < beta_M[d]:
            return 1

        return N_j[d-1] * chi(d-1)

    return N_j[dim-1] * chi(dim-1)


def sparse_grid(beta_M):
    # Max tensor-product grid
    M = np.prod(np.array(beta_M) + 1)
    N_M = level_to_grid(beta_M)
    x_M = get_grid(beta_M)
    L_M = np.prod(N_M)

    # Build index set of all possible betas
    betas = []
    beta_i = [np.arange(beta_M[d]+1) for d in range(len(beta_M))]
    for i, ele in enumerate(itertools.product(*beta_i)):
        betas.append(ele)

    # Print out info about each beta
    print(f'{"Beta":<15} {"N":<15} {"L_j":>5} {"G_j":>5} {"Gj2":>5} {"g_j":>5} {"Gamma_j"}')
    for beta_j in betas:
        G_j_size = group_size(beta_j, beta_M)
        N_j, L_j, G_j, g_j, gj_groups, gamma_j = get_beta_info(beta_j, x_M)
        print(f'{str(beta_j):<15} {str(N_j):<15} {L_j: 5d} {G_j: 5d} {G_j_size: 5d} {g_j: 5d} {gamma_j}')


if __name__ == '__main__':
    d = 4
    beta_M = (1, 1, 1)
    sparse_grid(beta_M)
