# -*- coding: utf-8 -*-

import json
import numpy as np
from scipy.special import erfi,erf,erfinv

def current_density_model(input_filename='j_input.json', output_filename='j_output.json'):
    f = open(input_filename)
    inputs = json.load(f)
    
    P_B = inputs['P_B']
    I_B0 = inputs['I_B0']
    r = inputs['r']
    alpha = inputs['alpha']
    f.close()
    
    theta=[0.747388,0.348462,-10.6654, 33.3149, 3.55288,0.142711]
    alpha=alpha*np.pi/180
    
    
    n=theta[4]*P_B+theta[5]
    alpha1=theta[1]*(theta[2]*P_B+theta[3])*np.pi/180
    alpha2=(theta[2]*P_B+theta[3])*np.pi/180
    ff=0.192754
    
    A1=(1-theta[0])/((np.pi**(3/2))/2*theta[1]*np.exp(-(theta[1]/2)**2)*(2*erfi(theta[1]/2)+erfi((np.pi*1j-(theta[1]**2))/(2*theta[1]))-erfi((np.pi*1j+(theta[1]**2))/(2*theta[1]))))
    A2=theta[0]/((np.pi**(3/2))/2*theta[2]*np.exp(-(theta[2]/2)**2)*(2*erfi(theta[2]/2)+erfi((np.pi*1j-(theta[2]**2))/(2*theta[2]))-erfi((np.pi*1j+(theta[2]**2))/(2*theta[2]))))
    jb=A1*np.exp(-(alpha/alpha1)**2)
    jscat=A2*np.exp(-(alpha/alpha2)**2)
    
    jmain=0.1*(I_B0/(r**2))*np.exp(-ff*n)*(jb+jscat)
    
    j_cex=I_B0*(1-np.exp(-ff*n))/(2*np.pi*r**2)*.1
    j=jmain+j_cex

    output = {"j": j}
    j_obj=json.dump(output)
    with open(output_filename, "w") as outfile:
        outfile.write(j_obj)
    return j