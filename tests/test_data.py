"""Testing for loading experimental data."""

import hallmd.data
from hallmd.data import spt100


def test_spt100_macdonald2019():
    expdata = hallmd.data.load(spt100.macdonald2019())
    assert len(expdata.keys()) == 3
    for opcond, data in expdata.items():
        assert data.thrust_N is None
        assert data.discharge_current_A is not None
        assert data.ion_velocity_m_s is not None
        assert data.ion_velocity_coords_m is not None
        assert data.ion_current_density_A_m2 is None
        assert data.ion_current_density_coords_m is None
        assert data.cathode_coupling_voltage_V is None
        assert data.ion_current_density_radius_m is None
        assert opcond.anode_mass_flow_rate_kg_s == 5.16
        assert opcond.discharge_voltage_V == 300.0

    data = [expdata[cond] for cond in expdata.keys()]

    L = [[data_i.log_likelihood_of(data_j) for data_j in data] for data_i in data]

    for i in range(len(data)):
        for j in range(len(data)):
            # Log likelihoods should always be <= 1 since variance is large for this dataset
            assert L[i][j] <= 0

            # Since the data have the same variance, the likelihoods should be symmetric
            assert L[j][i] == L[i][j]

            # Self-likelihood should be larger than other likelihoods
            if i != j:
                assert L[i][j] < L[i][i]


def test_spt100_diamant2014():
    expdata_L3 = hallmd.data.load(spt100.diamant2014("L3"))
    expdata_aerospace = hallmd.data.load(spt100.diamant2014("aerospace"))
    expdata_all_explicit = hallmd.data.load(spt100.diamant2014(["L3", "aerospace"]))
    expdata_all_implicit = hallmd.data.load(spt100.diamant2014())

    assert len(expdata_all_explicit.keys()) == len(expdata_all_implicit.keys())
    assert len(expdata_L3.keys()) + len(expdata_aerospace.keys()) == len(expdata_all_explicit.keys())

    for opcond, data in expdata_all_implicit.items():
        assert data.thrust_N is not None
        assert data.cathode_coupling_voltage_V is not None
        assert data.discharge_current_A is not None
        assert data.ion_velocity_m_s is None
        assert data.ion_velocity_coords_m is None
        if opcond in expdata_L3.keys():
            assert data.ion_current_density_A_m2 is not None
            assert data.ion_current_density_coords_m is not None
            assert data.ion_current_density_radius_m is not None
        else:
            assert data.ion_current_density_A_m2 is None
            assert data.ion_current_density_coords_m is None
            assert data.ion_current_density_radius_m is None

    data = [expdata_all_explicit[cond] for cond in expdata_all_explicit.keys()]

    L = [[data_i.log_likelihood_of(data_j) for data_j in data] for data_i in data]

    for i in range(len(data)):
        for j in range(len(data)):
            # Self-likelihood should be larger than other likelihoods
            if i != j:
                assert L[i][j] < L[i][i]

    # check that jion is being properly included in likelihood
    data_l3 = expdata_L3[list(expdata_L3.keys())[0]]

    L1 = data_l3.log_likelihood_of(data_l3)

    data_l3.ion_current_density_radius_m = None

    L3 = data_l3.log_likelihood_of(data_l3)

    assert L1 != L3


def test_spt100_sankovic1993():
    expdata = hallmd.data.load(spt100.sankovic1993())

    assert (len(expdata.keys())) > 100

    for opcond, data in expdata.items():
        assert data.thrust_N is not None
        assert data.discharge_current_A is not None
        assert data.cathode_coupling_voltage_V is None
        assert data.ion_velocity_m_s is None
        assert data.ion_velocity_coords_m is None
        assert data.ion_current_density_coords_m is None
        assert data.ion_current_density_A_m2 is None
        assert data.ion_current_density_radius_m is None
