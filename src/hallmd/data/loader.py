""" `loader.py`

Module for loading experimental data for specific thrusters.

Includes
--------
- `spt100_data()` - loads data for the SPT-100

!!! Info "Loading raw data"
    Raw data is loaded as best as possible into a standard format from any raw data source (.csv, .json, .hdf, etc.).
    The raw data files are included where possible along with wrapper Python `dataloader` functions that manage prepping
    the standard format. When adding new data to the repository, create a new folder for each thruster and always
    include corresponding `dataloader.py` files. You can then add top-level loaders for thrusters like `spt100_data()`
    here.

!!! Note "The standard data format"
    Data from a single experiment is loaded as best as possible into a Python `dict` with four fields:
    `[x, y, loc, var_y]` as explained in the example below. The experimental operating conditions `x` should be framed
    in the same units and format as any of your models would expect. Likewise, the measurements `y` should be
    directly comparable to the predictions of your models. `loc` is optional, but should be used when your data has
    spatial or temporal dependence (use `loc` in these cases to specify the Cartesian, spherical, etc. coordinates
    where the measurements were taken). `var_y` summarizes experimental noise in terms of additive Gaussian white noise
    with this variance.
    ```python
    data = dict(x=x,  # `(N, x_dim)` `np.ndarray` with `x_dim` experimental operating conditions for `N` data points
                y=y,  # `(N, y_dim)` `np.ndarray` with measurements of `y_dim` QoIs, corresponding to the `N` samples
                loc=loc,      # `(N, loc_dim)`, array with the `loc_dim` coordinates where the QoIs were measured
                var_y=var_y,  # `(N, y_dim)`, array with the experimental noise variance for all measurements
                )
    ```
"""
from hallmd import ExpData


def spt100_data(qois: list[str] = None) -> dict[str: list[ExpData]]:
    """Return a dict with experimental data for each specified quantity for the SPT-100.

    :param qois: a list specifying the experimental data to return, must be in `['V_cc', 'T', 'uion', 'jion']`
    :returns: map of `qoi->data`, where `data` is a list of experimental data sets
    """
    if qois is None:
        qois = ['V_cc', 'T', 'uion', 'jion', 'I_D']
    exp_data = dict()

    # Load Vcc data
    if 'V_cc' in qois:
        from .spt100.diamant2014.dataloader import load_vcc
        exp_data['V_cc'] = [load_vcc()]

    # Load thrust data
    if 'T' in qois:
        from .spt100.diamant2014.dataloader import load_thrust as thrust1
        from .spt100.sankovic1993.dataloader import load_thrust as thrust2
        exp_data['T'] = [thrust1(), thrust2()]

    # Load discharge current data
    if 'I_D' in qois:
        from .spt100.sankovic1993.dataloader import load_discharge_current
        exp_data['I_D'] = [load_discharge_current()]

    # Load ion velocity data
    if 'uion' in qois:
        from .spt100.macdonald2019.dataloader import load_uion
        exp_data['uion'] = [load_uion()]

    # Load ion velocity data
    if 'jion' in qois:
        from .spt100.diamant2014.dataloader import load_jion
        exp_data['jion'] = [load_jion()]

    return exp_data

def h9_data(qois: list[str] = None) -> dict[str: list[ExpData]]:
    """Return a dict with experimental data for each specified quantity for the H9.

    :param qois: a list specifying the experimental data to return, must be in `['V_cc', 'T', 'uion', 'jion']`
    :returns: map of `qoi->data`, where `data` is a list of experimental data sets
    """
    import sys
    import os
    secure_path = '/home/morag/h9-data'
    if os.path.exists(secure_path):
        sys.path.insert(0, secure_path)
    else:
        raise ImportError(f"The specified secure path does not exist: {secure_path}")
    try:
        import h9dataloader
    except ImportError:
        raise ImportError('Could not import ITAR data loader. Ensure the secure path is correct.')

    if qois is None:
        qois = ['V_cc', 'uion', 'jion']
    exp_data = dict()

    # Load Vcc data
    if 'V_cc' in qois:
        from h9dataloader import load_vcc
        exp_data['V_cc'] = [load_vcc()]

    # Load thrust data
    if 'T' in qois:
        print("No Thrust Data")
        # from h9dataloader import load_thrust
        # exp_data['T'] = [load_thrust()]

    # Load discharge current data
    if 'I_D' in qois:
        print("No I_D data")
        # from h9dataloader import load_discharge_current
        # exp_data['I_D'] = [load_discharge_current()]

    # Load ion velocity data
    if 'uion' in qois:
        from h9dataloader import load_uion
        exp_data['uion'] = [load_uion()]

    # Load ion velocity data
    if 'jion' in qois:
        from h9dataloader import load_jion
        exp_data['jion'] = [load_jion()]

    if 'GT' in qois:
        from h9dataloader import load_gt_jion, load_gt_thrust
        exp_data['gt_jion'] = [load_gt_jion()]
        exp_data['gt_thrust'] = [load_gt_thrust()]

    return exp_data

if __name__ == '__main__':
    data = h9_data(['V_cc', 'uion', 'jion', 'GT'])
    print(data)
