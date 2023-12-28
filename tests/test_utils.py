"""Testing for package utilities."""
import tempfile
import os

import matplotlib.pyplot as plt
import numpy as np

from hallmd.utils import data_write, ModelRunException, plot_qoi


def test_exception():
    try:
        raise ModelRunException('Testing an exception')
    except ModelRunException as e:
        assert True


def test_write():
    with tempfile.NamedTemporaryFile(suffix='.json', encoding='utf-8', mode='w', delete=False, dir='.') as fd:
        pass
    data = {'help': 'me'}
    data_write(data, fd.name)
    os.unlink(fd.name)


def test_plot():
    x = np.linspace(0, 1, 100).reshape((100, 1))
    qoi = x + (np.random.rand(100, 50)*0.2 - 0.1)
    fig, ax = plt.subplots()
    plot_qoi(ax, np.squeeze(x), qoi, 'X direction', 'Y direction', legend=False)
