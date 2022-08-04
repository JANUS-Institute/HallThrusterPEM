import json
import math

def cathode_coupling_model(input_filename='cc_input.json', output_filename='cc_output.json'):
    f = open(input_filename)
    inputs = json.load(f)
    Te = inputs['Te']
    PB = inputs['PB_norm']
    PT = inputs['PT_norm']
    Pstar = inputs['Pstar_norm']
    V_vac = inputs['V_vac']
    f.close()

    # Equation 12 in Jorns and Byrne, Plasma Sources Sci. Technol. 30 (2021) 015012
    V_cc = V_vac + Te * math.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB

    output = {"V_cc": V_cc}
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    return V_cc
