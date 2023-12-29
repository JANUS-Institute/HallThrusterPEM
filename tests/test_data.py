"""Testing for loading experimental data."""
from hallmd.data.loader import spt100_data


def test_spt100():
    data = spt100_data(['V_cc', 'T', 'jion', 'uion'])
    v_cc = data['V_cc'][0]
    assert v_cc['x'].shape[0] == v_cc['y'].shape[0] == v_cc['var_y'].shape[0]
