import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import pandas as pd

from WIMpy import DMUtils as DMU
#from lindhard import *
from lindhard import lindhard_transform 

def sigma_res(E, sigma_0 = 4e-3):
    """ Calculates the ionization energy resolution of the cluster reconstruction.
    """
    return np.sqrt(sigma_0**2+3.77e-3*0.133*E)
    #return sigma_0

def resolution(E1, E2, sigma_E):
    """ Gaussian resolution around E2 energy.
    """
    ## E1 is the energy point of the gaussian.
    ## E2 is the center energy of the gaussian.
    ## sigma_E cluster resolution
    deltaE = (E1.T-E2)
    return ((2*np.pi*sigma_E**2)**-0.5*np.exp(-0.5*(deltaE)**2/sigma_E**2)).T

def dR_dEe(Er, N_p_Si, N_n_Si, m_x, sigma_p, sigma_res_Ee = 4e-4, sigma_Ee = 4e-4, eff = 1, step = 200, n_std = 5, Ermin=1e-3, Ermax=15):
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
    l = lindhard_transform(Ermin, Ermax, step)
    ### Transform to ionization energy
    Ee = l.lindhard(Er)
    ### Calculate the resolution in both energy units
    #sigma_res_Ee_full = sigma_res(Ee, sigma_res_Ee) 
    #sigma_res_Er_full = l.lindhard_inv(sigma_res_Ee_full)#/l.lindhard_derivative(Er)
    ## Needs to be multiplied by dEnr/dEee
    #sigma_res_Er_full = sigma_res_Ee_full/l.lindhard_derivative(Er)
    ##
    #plt.plot(Er, sigma_res_Er_full, label="Differential")
    #plt.plot(Er, l.lindhard_inv(sigma_res_Ee_full), label="Direct")
    #plt.legend(loc="best")
    #plt.show()

    ## Integral limits in energy recoil units
    #Emin, Emax = np.clip(Er-l.lindhard_inv(n_std*sigma_res_Ee_full),0.04,1000), np.clip(Er+l.lindhard_inv(n_std*sigma_res_Ee_full),0.04,1000)
    ### Sigma region of energies for each point in Er
    #E_space = np.geomspace(Emin, Emax, step).T
    ### Calculates the integrand for each point
    ### No res
    if sigma_res_Ee is not None:
        ### Calculate the resolution in both energy units
        sigma_res_Ee_full = sigma_res(Ee, sigma_res_Ee) 
        sigma_res_Er_full = l.lindhard_inv(sigma_res_Ee_full)#/l.lindhard_derivative(Er)
        ## Needs to be multiplied by dEnr/dEee
        #sigma_res_Er_full = sigma_res_Ee_full/l.lindhard_derivative(Er)
        ## Integral limits in energy recoil units
        Emin, Emax = np.clip(Er-l.lindhard_inv(n_std*sigma_res_Ee_full),0.04,1000), np.clip(Er+l.lindhard_inv(n_std*sigma_res_Ee_full),0.04,1000)
        ### Sigma region of energies for each point in Er
        E_space = np.geomspace(Emin, Emax, step).T
        integrand = DMU.dRdE_standard(E_space, N_p_Si, N_n_Si, m_x, sigma_p)/l.lindhard_derivative(E_space)*resolution(l.lindhard(E_space),Ee,sigma_res_Ee_full)
        #plt.plot(l.lindhard(E_space), integrand)
        #plt.loglog()
        #plt.show()

    ### Integrates each point over its sigma interval
    if type(eff) is not float and type(eff) is not int and type(eff) is not float:
        ### If eff is the file name, interpolates the values of the file.
        eff = eff(Ee)
        ## No res
        if sigma_res_Ee is None:
            dR = eff*DMU.dRdE_standard(Er, N_p_Si, N_n_Si, m_x, sigma_p)/l.lindhard_derivative(Er)
        else:
            dR = eff*simpson(integrand,l.lindhard(E_space))
    else:
        ### Efficiency cut
        sigma_Er = l.lindhard_inv(sigma_Ee)
        #dR = eff*simpson(integrand,l.lindhard(E_space))
        ## No res
        dR = eff*DMU.dRdE_standard(Er, N_p_Si, N_n_Si, m_x, sigma_p)/l.lindhard_derivative(Er)
        dR[Er<=sigma_Er] = np.zeros([len(Er[Er<=sigma_Er])])

    #plt.plot(Ee, dR)
    #plt.plot(Ee, DMU.dRdE_standard(Er, N_p_Si, N_n_Si, m_x, sigma_p)/l.lindhard_derivative(Er))
    #plt.loglog()
    #plt.show()
    ### Efficiency cut
    #sigma_Er = l.lindhard_inv(sigma_Ee)
    #dR[Er<=sigma_Er] = np.zeros([len(Er[Er<=sigma_Er])])

    ### If it was originally a float returns a np.float
    if original_type:
        dR = np.float64(dR)
    return dR
