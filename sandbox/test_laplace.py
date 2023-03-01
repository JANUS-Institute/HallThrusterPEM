import numpy as np
import sys

sys.path.append('..')

from utils import approx_hess


def func(theta):
    theta = np.atleast_1d(theta)
    x1, x2, x3 = theta
    return x1 + x1*x2 + 3 * x2**2 * x3 + 3*x3**2


def func_hess(map):
    x1, x2, x3 = map
    H = np.zeros((3, 3))
    H[0, :] = np.array([0, 1, 0])
    H[1, :] = np.array([1, 6*x3, 6*x2])
    H[2, :] = np.array([0, 6*x2, 6])
    return H


def test_hess():
    N = 1000
    lb = -1000
    ub = 1000
    tol = 1e-4
    passed = True
    for i in range(N):
        theta = np.random.rand(3)*(ub-lb) + lb
        hess_exact = func_hess(theta)
        hess_approx = approx_hess(func, theta, pert=0.01)
        res = np.abs(hess_exact - hess_approx)
        if np.any(res > tol):
            passed = False
            print(f'Not good within tolerance: {np.max(res)} > {tol}')

    if passed:
        print('All test cases passed')


if __name__ == '__main__':
    test_hess()
