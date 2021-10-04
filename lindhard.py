import numpy as np
from pynverse import inversefunc
from scipy.misc import derivative

##To avoid RuntimeWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def lindhard(E):
    """ Applies the conversion of Energy nuclear recoils to the ionization energy measured by the CCD.
    """
    if type(E) is list:
        E = np.array(E)
    ep = E*10.5/np.power(14,7/3)
    g = 0.0100686*np.power(ep,-0.02907) + 0.0100686*np.power(ep,-0.02907) + ep
    fact = 2.81961*g/(1.+2.81961*g)
    return E*fact


def lindhard_inv(Ee):
    """ Computes the numercial inverse function of the lindhard model. This converts ionization energy to nuclear recoils.
    """
    return inversefunc(lindhard, y_values = Ee).astype(float)


def lindhard_derivative(E, dx=1e-6):
    """ Calculates the dEe/dEr useful to some calculus for the differential rate.
    """
    if type(E) is list:
        E = np.array(E)
    return derivative(lindhard, E, dx=dx)



