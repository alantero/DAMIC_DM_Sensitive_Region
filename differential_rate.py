import numpy as np
from scipy.integrate import simpson


from WIMpy import DMUtils as DMU
from lindhard import *

def sigma_res(E, sigma_0 = 4e-3):
    """ Calculates the ionization energy resolution of the cluster reconstruction.
    """
    return np.sqrt(sigma_0**2+3.77e-3*0.133*E)


def resolution(E1, E2, sigma_E):
    """ Gaussian resolution around E2 energy.
    """
    ## E1 is the energy point of the gaussian.
    ## E2 is the center energy of the gaussian.
    ## sigma_E cluster resolution
    deltaE = (E1.T-E2)
    return ((2*np.pi*sigma_E**2)**-0.5*np.exp(-0.5*(deltaE)**2/sigma_E**2)).T



def dR_dEe(Er, N_p_Si, N_n_Si, m_x, sigma_p, sigma_Ee = 4e-3):
    """ Calculates the rate at the energy point Er.
        Er must have nuclear recoil units.
        The integral is performed in nuclear recoil units.
    """
    ### Conversion of lists or float to numpy array
    original_type = False
    if type(Er) is not np.ndarray:
        if type(Er) is not list:
            original_type = True
            Er = [Er]
        Er = np.array(Er)
    ### Transform to ionization energy
    Ee = lindhard(Er)
    ### Calculate the resolution in both energy units
    sigma_Ee = sigma_res(Ee, sigma_Ee)  
    sigma_Er = lindhard_inv(sigma_Ee)

    ## Integral limits in energy recoil units
    n_std, step = 5, 100
    Emin, Emax = np.clip(Er-n_std*sigma_Er,1e-5,step), np.clip(Er+n_std*sigma_Er,1e-5,step)
    ### Sigma region of energies for each point in Er
    E_space = np.geomspace(Emin, Emax, step).T
    ### Calculates the integrand for each point
    integrand = DMU.dRdE_standard(E_space, N_p_Si, N_n_Si, m_x, sigma_p)/lindhard_derivative(E_space)*resolution(lindhard(E_space),Ee,sigma_Ee)

    ### Integrates each point over its sigma interval
    eff = 0.75

    dR = eff*simpson(integrand,lindhard(E_space))
    ### Efficiency cut
    dR[Er<=sigma_Ee] = np.zeros([len(Er[Er<=sigma_Ee])])

    ### If it was originally a float returns a np.float
    if original_type:
        dR = np.float64(dR)
    return dR
