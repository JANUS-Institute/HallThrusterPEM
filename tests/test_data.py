"""Testing for loading and handling experimental data."""

import numpy as np

import hallmd.data
from hallmd.data import OperatingCondition, spt100


def test_thrusterdata():
    """Test converting amisc pem data into ThrusterData objects."""
    mdots = [5e-6, 6e-6]
    pbs = [1e-6, 10e-6]
    vas = [300.0, 400.0]
    vccs = [20.0, 30.0]
    Ts = [0.2, 0.3]
    Ids = [6.0, 7.0]
    Ibs = [4.0, 6.0]
    uion_coords = np.linspace(0.0, 0.05, 100)
    uions = [uion_coords**2, 2 * uion_coords**2]
    jion_coords = np.linspace(0.0, 0.5 * np.pi, 15)
    jions = [3 * np.exp(-jion_coords), 2 * np.exp(-jion_coords), np.exp(-jion_coords)]
    radii = [1.0, 2.0, 3.0]

    operating_conditions = [
        OperatingCondition(anode_mass_flow_rate_kg_s=mdot, background_pressure_torr=pb, discharge_voltage_v=va)
        for (mdot, pb, va) in zip(mdots, pbs, vas)
    ]

    jion_tensor = np.zeros((len(operating_conditions), len(jion_coords), len(radii)))
    for i, _ in enumerate(operating_conditions):
        for j, jion in enumerate(jions):
            jion_tensor[i, :, j] = i * jion

    pem_dict = {
        "T": Ts,
        "V_cc": vccs,
        "I_d": Ids,
        "I_B0": Ibs,
        "u_ion_coords": np.array([uion_coords for _ in operating_conditions]),
        "u_ion": uions,
        "j_ion_coords": np.array([jion_coords for _ in operating_conditions]),
        "j_ion": jion_tensor,
        "eta_m": np.zeros(len(operating_conditions)),
        "eta_v": np.zeros(len(operating_conditions)),
        "eta_c": np.zeros(len(operating_conditions)),
        "eta_a": np.zeros(len(operating_conditions)),
    }

    out = hallmd.data.pem_to_thrusterdata(
        operating_conditions, pem_dict, sweep_radii=np.array(radii), use_corrected_thrust=False
    )

    for i, (opcond, tdata) in enumerate(out.items()):
        assert opcond == operating_conditions[i]

        assert tdata.thrust_N is not None and tdata.thrust_N.mean == Ts[i]
        assert tdata.discharge_current_A is not None and tdata.discharge_current_A.mean == Ids[i]
        assert tdata.ion_current_A is not None and tdata.ion_current_A.mean == Ibs[i]
        assert tdata.cathode_coupling_voltage_V is not None and tdata.cathode_coupling_voltage_V.mean == vccs[i]

        assert tdata.ion_velocity is not None
        assert np.all(tdata.ion_velocity.axial_distance_m == uion_coords)
        assert np.all(tdata.ion_velocity.velocity_m_s.mean == uions[i])

        assert tdata.ion_current_sweeps is not None and len(tdata.ion_current_sweeps) == len(radii)
        for j, sweep in enumerate(tdata.ion_current_sweeps):
            assert sweep.radius_m == radii[j]
            assert np.all(sweep.angles_rad == jion_coords)
            assert np.all(sweep.current_density_A_m2.mean == i * jions[j])


def test_spt100_macdonald2019():
    expdata = hallmd.data.load(spt100._macdonald2019())
    assert len(expdata.keys()) == 3
    for opcond, data in expdata.items():
        assert data.thrust_N is None
        assert data.discharge_current_A is not None
        assert data.ion_velocity is not None
        assert data.ion_current_sweeps is None
        assert data.cathode_coupling_voltage_V is None
        assert opcond.anode_mass_flow_rate_kg_s == 5.16e-6
        assert opcond.discharge_voltage_v == 300.0


def test_spt100_diamant2014():
    expdata_L3 = hallmd.data.load(spt100._diamant2014("L3"))
    expdata_aerospace = hallmd.data.load(spt100._diamant2014("aerospace"))
    expdata_all_explicit = hallmd.data.load(spt100._diamant2014(["L3", "aerospace"]))
    expdata_all_implicit = hallmd.data.load(spt100._diamant2014())

    assert len(expdata_all_explicit.keys()) == len(expdata_all_implicit.keys())
    assert len(expdata_L3.keys()) + len(expdata_aerospace.keys()) == len(expdata_all_explicit.keys())
    assert len(expdata_all_explicit.keys()) == 15

    for opcond, data in expdata_all_implicit.items():
        assert data.thrust_N is not None
        assert data.cathode_coupling_voltage_V is not None
        assert data.discharge_current_A is not None
        assert data.ion_velocity is None
        if opcond in expdata_L3.keys():
            assert data.ion_current_sweeps is not None
        else:
            assert data.ion_current_sweeps is None is None


def test_spt100_sankovic1993():
    expdata = hallmd.data.load(spt100._sankovic1993())

    assert (len(expdata.keys())) > 100

    for opcond, data in expdata.items():
        assert data.thrust_N is not None
        assert data.discharge_current_A is not None
        assert data.cathode_coupling_voltage_V is None
        assert data.ion_velocity is None
        assert data.ion_current_sweeps is None


# Removing this for now since it doesn't pass locally when we actually have the H9
# and if this ran on CI and failed then we've committed the H9 data to a public repo
# and wouldn't be able to do much
# def test_h9_empty():
#     thruster = hallmd.data.get_thruster('H9')
#
#     with pytest.raises(ImportError):
#         _ = thruster.all_data()
