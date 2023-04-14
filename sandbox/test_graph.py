import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def rand_dag(Nk):
    """Generate a random DAG graph of size Nk"""
    adj = np.random.randint(0, 2, (Nk, Nk), dtype=int)
    np.fill_diagonal(adj, 0)
    adj *= np.triu(adj)

    # Make sure there is at least 1 edge in each row
    for k in range(Nk-1):
        if not np.any(adj[k, k+1:]):
            idx = np.random.randint(k+1, Nk)
            adj[k, idx] = 1

    return adj


def add_cycles(adj, N):
    """Add N backward edges in the adjacency matrix (will form cycles)"""
    Nk = adj.shape[0]
    cnt = 0
    while cnt < N:
        for k in range(1, Nk):
            idx = np.random.randint(0, k)
            if np.random.rand() < 0.5 and adj[k, idx] == 0:
                adj[k, idx] = 1
                cnt += 1
                if cnt >= N:
                    break


def test_graph():
    """Just testing out the networkx library, and some stuff with SCCs"""
    # Construct and show graph
    Nk = 8
    adj = rand_dag(Nk)
    add_cycles(adj, 3)
    print('Adjacency matrix:')
    print(adj)
    g = nx.DiGraph(adj)
    nx.draw(g, with_labels=True)
    plt.show()

    # Condense the graph into a DAG of SCCs
    c = nx.condensation(g)
    print('\nTopological ordering of condensation:')
    for node in nx.topological_sort(c):
        pre = [n for n in c.predecessors(node)]
        group = [n for n in c.nodes[node]['members']]
        print(f'Current super-node: {node}. SCC members: {group}. Predecessors: {pre}.')


if __name__ == '__main__':
    test_graph()
