import numpy as np 
from scipy.integrate import simpson
from pynverse import inversefunc
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from WIMpy import DMUtils as DMU
from lindhard import lindhard_transform
from differential_rate import *


def random_signal_det(n_size, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection, Eemin = 4e-10, Eemax = 7, sigma_res_Ee = 4e-4, sigma_Ee = 4e-4, eff = 1, step = 400):

    l = lindhard_transform(0.04, Eemax*2, step)
    events = []
    
    if detection:
        ### To simplify the function call
        dRdE_lamb = lambda Er: dR_dEe(Er, N_p_Si, N_n_Si, m_x, sigma_p, sigma_res_Ee, sigma_Ee, eff=eff, step=step, Ermin=l.lindhard_inv(Eemin), Ermax=l.lindhard_inv(Eemax))
    else:
        dRdE_lamb = lambda Er: DMU.dRdE_standard(Er, N_p_Si, N_n_Si, m_x, sigma_p)

    ### We define fs in all the energy region
    Ermax = l.lindhard_inv(Eemax)
    Er_space = np.geomspace(l.lindhard_inv(Eemin),Ermax,step)
    dRdE_space = dRdE_lamb(Er_space)
        
    ### We search the energy where dRdE < 1e-4
    ### For low masses this will reduce the computation time
    ### As the dRdE is non-negligible only in a short range of energies.
    if np.sum(dRdE_space <= 1e-4) > 0:
        w = np.where(dRdE_space <= 1e-4)[0]
        ### We avoid the first element of the array
        ### We choose the next number near the limit
        ### To make sure we dont cut to much distribution
        Ermax = Er_space[np.min(w[w!=0])+1]
            
        ### We redefine the energy space
        Er_space = np.geomspace(l.lindhard_inv(Eemin),Ermax,step)
        dRdE_space = dRdE_lamb(Er_space)


    ### We generate random points in x->[Ermin,Ermax] 
    ### and y -> [1e-4, drdE_max]
    ### If the point is inside the curve it is an event.
    ### We do this until reaching the desired number of events
    while(len(events)<n_size):
        x,y = np.random.uniform(l.lindhard_inv(sigma_Ee),Ermax), np.random.uniform(1e-4,np.max(dRdE_space/C_sigma0))
        if y <= dRdE_lamb(x)/C_sigma0:
            events.append(x)

    if detection:
        events = l.lindhard(events).tolist()
    return events


def random_background(n_size, Eemin, Eemax):
    """ For consistency we give Ermin and Ermax in Enr units.
    """
    E_bkg = np.random.uniform(Eemin, Eemax, n_size)
    return E_bkg.tolist()

