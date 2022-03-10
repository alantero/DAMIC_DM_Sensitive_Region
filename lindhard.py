import numpy as np
from scipy.interpolate import UnivariateSpline

class lindhard_transform(object):
    def __init__(self, Ermin, Ermax, step = 200):
        self.Ermin, self.Ermax = Ermin, Ermax
        self.step = step
        self.Er = np.geomspace(self.Ermin, self.Ermax, self.step)
        self.interpol_inv = UnivariateSpline(self.lindhard(self.Er), self.Er, k=1, s=0)
        self.interpol_der = UnivariateSpline(self.Er, self.lindhard(self.Er), k=1, s=0).derivative()

    def lindhard(self, E):
        """ Applies the conversion of Energy nuclear recoils to the ionization energy measured by the CCD.
        """
        if type(E) is list:
            E = np.array(E)
        ep = E*10.5/np.power(14,7/3)
        g = 0.0100686*np.power(ep,-0.02907) + 0.0100686*np.power(ep,-0.02907) + ep
        fact = 2.81961*g/(1.+2.81961*g)
        return E*fact

    def lindhard_inv(self, Ee):
        """ Computes the numercial inverse function of the lindhard model. This converts ionization energy to nuclear recoils.
        """
        return self.interpol_inv(Ee)

    def lindhard_derivative(self, Er):#dx=1e-6):
        """ Calculates the dEe/dEr useful to some calculus for the differential rate.
        """
        if type(Er) is list:
            Er = np.array(Er)
        return self.interpol_der(Er)
