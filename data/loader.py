def spt100_data(qois=None):
    """Return a dict with experimental data for each specified quantity for the SPT-100
    :param qois: a list() of str specifying the experimental data to return, must be in ['V_cc', 'T', 'uion', 'jion']
    """
    if qois is None:
        qois = ['V_cc', 'T', 'uion', 'jion']
    exp_data = {qoi: None for qoi in qois}

    # Load Vcc data
    if 'V_cc' in exp_data:
        from data.spt100.diamant2014.dataloader import load_vcc
        exp_data['V_cc'] = [load_vcc()]

    # Load thrust data
    if 'T' in exp_data:
        from data.spt100.diamant2014.dataloader import load_thrust as thrust1
        from data.spt100.sankovic1993.dataloader import load_thrust as thrust2
        exp_data['T'] = [thrust1(), thrust2()]

    # Load ion velocity data
    if 'uion' in exp_data:
        from data.spt100.macdonald2019.dataloader import load_uion
        exp_data['uion'] = [load_uion()]

    # Load ion velocity data
    if 'jion' in exp_data:
        from data.spt100.diamant2014.dataloader import load_jion
        exp_data['jion'] = [load_jion()]

    return exp_data
