import numpy as np 
from scipy.integrate import simpson
from pynverse import inversefunc
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline


from WIMpy import DMUtils as DMU
from lindhard import *
from differential_rate import *

import matplotlib.pyplot as plt

def cdf_signal(Er, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection=False, sigma_Ee = 4e-3):  
    """ Calculates the cumulative distribution function at value Er.
        It calculates it for the theoretical differential rate or the detected.
    """
    ndarr=isinstance(Er, (np.ndarray))
    if ndarr:
        ### Por la cara a veces con random signal llega un ndarray
        Er = Er[0]

    if  Er > lindhard_inv(sigma_Ee):
        if detection:
            E_space = np.geomspace(lindhard_inv(sigma_Ee),Er,300)
            fs = dR_dEe(E_space,N_p_Si, N_n_Si, m_x, sigma_p)/C_sigma0
            cdf = simpson(fs, lindhard(E_space))
        else:
            E_space = np.geomspace(lindhard_inv(sigma_Ee),Er,300)
            cdf = simpson(DMU.dRdE_standard(E_space, N_p_Si, N_n_Si, m_x, sigma_p)/C_sigma0, E_space)
    else:
        cdf = 0.0
    
    if ndarr:
        cdf = np.array([cdf])
    return cdf




def random_signal(n_size, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, Emin = 4e-3, Emax = 7, detection=False):
    """ Simulates the energy of the events.
        Returns the energy in Nuclear recoil units.
    """
    u = np.random.rand(n_size)

    events = []
    if detection:
        dom = [Emin,Emax]    
    else:
        dom = [lindhard_inv(Emin),lindhard_inv(Emax)] 
    for x in u:
        E_event = float(inversefunc(cdf_signal, y_values = x, args=(N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection), domain=dom,accuracy=4))

        if E_event < 0:
            """ This problem is due to dRdEe is not a continuos function.
                When it reaches the efficiency threshold it fails to calculate the inverse.
                We just recalculate the random number until we obtain E_event>0.
            """
            negative = True
            while negative:
                u_i = np.random.rand(1)
                E_event = float(inversefunc(cdf_signal, y_values = u_i, args=(N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection)))
                if E_event>0:
                    negative = False
        events.append(E_event)

    return events


def random_signal_det(n_size, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, Eemin = 4e-3, Eemax = 7, step = 100):
    """ Simulates the energy of the detected events.
        Returns the energy in Nuclear recoils units.
    """

    def inv(E, y0):
        """ Calculates the inverse value of the cumulative distribution function.
            Finds the E that minimizes (cdf(E)-y0)**2. Which is the energy of the event.
        """
        return (cdf_signal(E, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection = True)-y0)**2

    u = np.random.rand(n_size)

    events = []

    bnds = [[lindhard_inv(Eemin),lindhard_inv(Eemax)] ]
    for x in u:
        E_event = minimize(inv, 1, args = (x), bounds=bnds,method="L-BFGS-B", tol=1e-6).x[0]
        events.append(E_event)

    return lindhard(events).tolist()



def random_background(n_size, Eemin, Eemax):
    """ For consistency we give Ermin and Ermax in Enr units.
    """
    E_bkg = np.random.uniform(Eemin, Eemax, n_size)
    return E_bkg.tolist()

