"""Testing for loading experimental data."""

import hallmd.data
from hallmd.data import spt100


def test_spt100_macdonald2019():
    expdata = hallmd.data.load(spt100.macdonald2019())
    assert len(expdata.keys()) == 3
    for opcond, data in expdata.items():
        assert data.T is None
        assert data.I_D is not None
        assert data.uion is not None
        assert data.uion_coords is not None
        assert data.jion is None
        assert data.jion_coords is None
        assert data.V_cc is None
        assert data.radius is None
        assert opcond.mdot_a == 5.16
        assert opcond.V_a == 300.0


def test_spt100_diamant2014():
    expdata_L3 = hallmd.data.load(spt100.diamant2014("L3"))
    expdata_aerospace = hallmd.data.load(spt100.diamant2014("aerospace"))
    expdata_all_explicit = hallmd.data.load(spt100.diamant2014(["L3", "aerospace"]))
    expdata_all_implicit = hallmd.data.load(spt100.diamant2014())

    assert len(expdata_all_explicit.keys()) == len(expdata_all_implicit.keys())
    assert len(expdata_L3.keys()) + len(expdata_aerospace.keys()) == len(expdata_all_explicit.keys())

    for opcond, data in expdata_all_implicit.items():
        assert data.T is not None
        assert data.V_cc is not None
        assert data.I_D is not None
        assert data.uion is None
        assert data.uion_coords is None
        if opcond in expdata_L3.keys():
            assert data.jion is not None
            assert data.jion_coords is not None
            assert data.radius is not None
        else:
            assert data.jion is None
            assert data.jion_coords is None
            assert data.radius is None


def test_spt100_sankovic1993():
    expdata = hallmd.data.load(spt100.sankovic1993())

    assert (len(expdata.keys())) > 100

    for opcond, data in expdata.items():
        assert data.T is not None
        assert data.I_D is not None
        assert data.V_cc is None
        assert data.uion is None
        assert data.uion_coords is None
        assert data.jion_coords is None
        assert data.jion is None
        assert data.radius is None
