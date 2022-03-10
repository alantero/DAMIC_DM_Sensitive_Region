import numpy as np 
from scipy.integrate import simpson
from pynverse import inversefunc
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from WIMpy import DMUtils as DMU
#from lindhard import *
from lindhard import lindhard_transform
from differential_rate import *


def cdf_signal(Er, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection=False, sigma_res_Ee = 4e-4, sigma_Ee = 4e-4,Eemin = 1e-10, Eemax = 7, eff = 1, step = 400):  
    """ Calculates the cumulative distribution function at value Er.
        It calculates it for the theoretical differential rate or the detected.
    """
    ndarr=isinstance(Er, (np.ndarray))
    if ndarr:
        ### Por la cara a veces con random signal llega un ndarray
        Er = Er[0]


    ### FIXME Quick tranform of Ee to Er to calculate lindhard
    l = lindhard_transform(Eemin*2, Eemax*2, step)
    l = lindhard_transform(l.lindhard_inv(Eemin), l.lindhard_inv(Eemax), step)

    if detection:
        sigma_Er = l.lindhard_inv(sigma_Ee)
        ### cdf below this threshold is 0
        if  Er > sigma_Er:
            ### The E_space must be all the space. NOT ONLY FROM sigma_Er to Er
            E_space = np.geomspace(l.lindhard_inv(Eemin),Er,step)
            fs = dR_dEe(E_space, N_p_Si, N_n_Si, m_x, sigma_p, sigma_res_Ee, sigma_Ee, eff=eff, step=step, Ermin=l.lindhard_inv(Eemin), Ermax=l.lindhard_inv(Eemax))/C_sigma0
            #E_ = np.geomspace(l.lindhard_inv(Eemin), l.lindhard_inv(Eemax), step)
            #fs_ = dR_dEe(E_, N_p_Si, N_n_Si, m_x, sigma_p, sigma_res_Ee, sigma_Ee, step)/C_sigma0
            #print(simpson(fs_,l.lindhard(E_)))
            cdf = simpson(fs, l.lindhard(E_space))
            #print(cdf)
        else:
            cdf = 0.0
    else:
        E_space = np.geomspace(l.lindhard_inv(Eemin),Er,step)
        cdf = simpson(DMU.dRdE_standard(E_space, N_p_Si, N_n_Si, m_x, sigma_p)/C_sigma0, E_space)
    #if ndarr:
    #    cdf = np.array([cdf])
    return cdf


#def random_signal(n_size, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, Emin = 4e-4, Emax = 7, detection=False):
#    """ Simulates the energy of the events.
#        Returns the energy in Nuclear recoil units.
#    """
#    u = np.random.rand(n_size)
#
#    events = []
#    if detection:
#        dom = [Emin,Emax]    
#    else:
#        dom = [lindhard_inv(Emin),lindhard_inv(Emax)] 
#    for x in u:
#        E_event = float(inversefunc(cdf_signal, y_values = x, args=(N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection), domain=dom,accuracy=4))

#        if E_event < 0:
#            """ This problem is due to dRdEe is not a continuos function.
#                When it reaches the efficiency threshold it fails to calculate the inverse.
#                We just recalculate the random number until we obtain E_event>0.
#            """
#            negative = True
#            while negative:
#                u_i = np.random.rand(1)
#                E_event = float(inversefunc(cdf_signal, y_values = u_i, args=(N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection)))
#                if E_event>0:
#                    negative = False
#        events.append(E_event)
#
#    return events


#def random_signal_det(n_size, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, Eemin = 4e-4, Eemax = 7, sigma_res_Ee = 4e-4, sigma_Ee = 4e-4, step = 100):
#    """ Simulates the energy of the detected events.
#        Returns the energy in Nuclear recoils units.
#    """
#
#    def inv(E, y0):
#        """ Calculates the inverse value of the cumulative distribution function.
#            Finds the E that minimizes (cdf(E)-y0)**2. Which is the energy of the event.
#        """
#        return (cdf_signal(E, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection = True, sigma_res_Ee=sigma_res_Ee, sigma_Ee=sigma_Ee)-y0)**2
#
#    u = np.random.rand(n_size)
#
#    events = []
#
#    bnds = [[lindhard_inv(Eemin),lindhard_inv(Eemax)] ]
#    for x in u:
#        E_event = minimize(inv, 1, args = (x), bounds=bnds,method="L-BFGS-B", tol=1e-6).x[0]
#        events.append(E_event)
#
#    return lindhard(events).tolist()

def random_signal_det(n_size, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection, Eemin = 4e-10, Eemax = 7, sigma_res_Ee = 4e-4, sigma_Ee = 4e-4, eff = 1, step = 400):
    """ Simulates the energy of the detected events.
        Returns the energy in Nuclear recoils units.
    """
    ### FIXME Quick tranform of Ee to Er to calculate lindhard
    l = lindhard_transform(Eemin*2, Eemax*2, step)

    if detection:
        cdf_thres = cdf_signal(sigma_Ee, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection, sigma_res_Ee=sigma_res_Ee, sigma_Ee=sigma_Ee, Eemin = Eemin, Eemax = Eemax, eff = eff, step=step)
    else:
        cdf_thres = 0
    u = np.random.uniform(cdf_thres, 1, n_size)
    #E_range = np.geomspace(lindhard_inv(Eemin), lindhard_inv(Eemax), step)
    E_range = np.geomspace(l.lindhard_inv(sigma_Ee), l.lindhard_inv(Eemax), step)
    #print(C_sigma0)
    #print(simpson(dR_dEe(E_range, N_p_Si, N_n_Si, m_x, sigma_p, sigma_res_Ee, sigma_Ee, step), lindhard(E_range)))
    cdf_range = np.array([cdf_signal(E, N_p_Si, N_n_Si, m_x, sigma_p, C_sigma0, detection, sigma_res_Ee=sigma_res_Ee, sigma_Ee=sigma_Ee, Eemin=Eemin, Eemax=Eemax, eff=eff, step=step) for E in E_range])
    #input()
    #plt.plot(E_range, cdf_range)
    cdf_uni, indices = np.unique(cdf_range[~np.isnan(cdf_range)], return_index=True)
    cdf_inv = UnivariateSpline(cdf_uni, E_range[~np.isnan(cdf_range)][indices], k = 1, s = 0)
    ### Checking plots
    #plt.plot(E_range, cdf_range)
    ##plt.plot(E_range[~np.isnan(cdf_range)][indices], cdf_uni, alpha = 0.7)
    #plt.show()
    #if detection:
    #    plt.plot(np.linspace(cdf_thres,1,100), lindhard(cdf_inv(np.linspace(cdf_thres,1,100))) )
    #    plt.show()
    #else:
    #    plt.plot(np.linspace(cdf_thres,1,100), cdf_inv(np.linspace(cdf_thres,1,100)) )
    #plt.show()

    #events = []
    #for x in u:
    #    #print("Event")
    #    #print(x)
    #    #print(cdf_inv(x))
    events = cdf_inv(u)    #events.append(cdf_inv(x))
    if detection:
        ### Prevent for numerical limitations to have events below sigma_Ee
        while len(events[l.lindhard(events)<=sigma_Ee]) != 0:
            events[l.lindhard(events)<=sigma_Ee] = cdf_inv( np.random.uniform(cdf_thres, 1, len(events[l.lindhard(events)<=sigma_Ee])) )
        return l.lindhard(events).tolist()
    else:
        return events.tolist()



def random_background(n_size, Eemin, Eemax):
    """ For consistency we give Ermin and Ermax in Enr units.
    """
    E_bkg = np.random.uniform(Eemin, Eemax, n_size)
    return E_bkg.tolist()

