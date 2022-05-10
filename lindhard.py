import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd
from scipy.interpolate import CubicSpline

class lindhard_transform(object):
    def __init__(self, Ermin, Ermax, step = 200):
        self.Ermin, self.Ermax = Ermin, Ermax
        self.step = step
        self.Er = np.geomspace(self.Ermin, self.Ermax, self.step)
        self.Ee = self.lindhard_real(self.Er,0.15,14)

        ### Region II 
        lindhard_CDMS = pd.read_csv("Lindhard_CDMS_Full.csv", comment="#")
        self.Er_CDMS,self.Ee_CDMS = lindhard_CDMS["Er"].to_numpy(),lindhard_CDMS["Ee"]*lindhard_CDMS["Er"].to_numpy()

        ### Region Measurements
        lindhard_CDMS = pd.read_csv("Lindhard_points.csv", comment="#")
        self.Er_p,self.Ee_p = lindhard_CDMS["Er"].to_numpy(),lindhard_CDMS["Ee"].to_numpy()
 
        ### Region Lindhard 
        self.lindhard_real_inv = CubicSpline(self.Ee, self.Er)
        self.lindhard_real_der = UnivariateSpline(self.Er, self.Ee, k=3, s=2).derivative()



    """
    #def lindhard(self, E):
    def lindhard_DAMIC(self, E):
        # Applies the conversion of Energy nuclear recoils to the ionization energy measured by the CCD.
        
        if type(E) is list:
            E = np.array(E)
        ep = E*10.5/np.power(14,7/3)
        g = 0.0100686*np.power(ep,-0.02907) + 0.0100686*np.power(ep,-0.02907) + ep
        fact = 2.81961*g/(1.+2.81961*g)
        return E*fact
    """

    def lindhard_real(self,E,k,Z):
        """ Lindhard parametrization for high energies.
        """
        cz = 11.5/np.power(Z,7/3)
        ep = E*cz
        g_E = 3*np.power(ep,0.15)+0.7*np.power(ep,0.6)+ep
        v = ep/(1+k*g_E)
        f = (ep - v)/ep
        return E*f

    def lindhard_points(self, func="Direct"):
        """ Ionization efficiency for the DAMIC Sb-Be measurements
        """
        #lindhard_CDMS = pd.read_csv("Lindhard_points.csv", comment="#")
        #Er,Ee = lindhard_CDMS["Er"],lindhard_CDMS["Ee"]
        if func == "Direct":
            cs = CubicSpline(self.Er_p,self.Ee_p)
        elif func == "Inv":
            cs = CubicSpline(self.Ee_p,self.Er_p)
        elif func == "Der":
            cs =  UnivariateSpline(self.Er_p,self.Ee_p,k=3,s=2).derivative()
        return cs

    def lindhard_splin(self, func="Direct"):
        """ Ionization efficiency in region 2.28 to 15 eVr
        """
        #lindhard_CDMS = pd.read_csv("Lindhard_CDMS_Full.csv", comment="#")
        #Er,Ee = lindhard_CDMS["Er"],lindhard_CDMS["Ee"]*lindhard_CDMS["Er"]
        #Er_s = self.Er_CDMS.loc[(self.Er_CDMS >= 2.28) & (self.Er_CDMS <= 15)]
        #Ee_s = self.Ee_CDMS.loc[(self.Er_CDMS >= 2.28) & (self.Er_CDMS <= 15)]
        Er_s = self.Er_CDMS[(self.Er_CDMS >= 2.28) & (self.Er_CDMS <= 15)]
        Ee_s = self.Ee_CDMS[(self.Er_CDMS >= 2.28) & (self.Er_CDMS <= 15)]
 
        if func == "Direct":
            cs = CubicSpline(Er_s,Ee_s)
        elif func == "Inv":
            cs = CubicSpline(Ee_s,Er_s)
        elif func == "Der":
            cs =  UnivariateSpline(Er_s,Ee_s,k=3,s=2).derivative()
        return cs

    def lindhard_low_thres(self, func="Direct"):
        """ Ionization efficiency below DAMIC measurements
        """
        #lindhard_CDMS = pd.read_csv("Lindhard_CDMS_Full.csv", comment="#")
        #Er,Ee = lindhard_CDMS["Er"],lindhard_CDMS["Ee"]*lindhard_CDMS["Er"]
        #Er_s = self.Er_CDMS.loc[(self.Er_CDMS >= 0.04) & (self.Er_CDMS <= 0.675)]
        #Ee_s = self.Ee_CDMS.loc[(self.Er_CDMS >= 0.04) & (self.Er_CDMS <= 0.675)]
        Er_s = self.Er_CDMS[(self.Er_CDMS >= 0.04) & (self.Er_CDMS <= 0.675)]
        Ee_s = self.Ee_CDMS[(self.Er_CDMS >= 0.04) & (self.Er_CDMS <= 0.675)]
 
        #cs = CubicSpline(Er_s,Ee_s)
        #cs =  UnivariateSpline(Er_s,Ee_s,k=3,s=None)
        #cs = np.poly1d(np.polyfit(Er_s,Ee_s,5))
        if func == "Direct":
            cs = CubicSpline(Er_s,Ee_s)
        elif func == "Inv":
            cs = CubicSpline(Ee_s,Er_s)
        elif func == "Der":
            cs =  UnivariateSpline(Er_s,Ee_s,k=3,s=2).derivative()
        return cs

    def lindhard(self,E):
        if type(E) is list:
            E = np.array(E)

        if type(E) is np.ndarray:
            Ee = np.zeros(E.shape)
            ### Region I
            Ee[E<=0.04] = [0] * len([E<=0.04])
            ### Region II
            Ee[(E>0.04) & (E<0.675)] = self.lindhard_low_thres()(E[(E>0.04) & (E<0.675)])
            ### Region III (DAMIC Fit)
            Ee[(E>=0.675) & (E<=2.28)] = self.lindhard_points()(E[(E>=0.675) & (E<=2.28)])
            ### Region IV (Seconds interpolation DAMIC Fit)
            Ee[(E>2.28) & (E<15)] = self.lindhard_splin()(E[(E>2.28) & (E<15)])
            ### Region V (Lindhard k=0.15)
            Ee[E>=15] = self.lindhard_real(E[E>=15],0.15,14)
        else:
            if E<=0.04: Ee = 0
            if E>0.04 and E<0.675: Ee = self.lindhard_low_thres()(E)
            if E>=0.675 and E<=2.28: Ee = self.lindhard_points()(E)
            if E>2.28 and E<15: Ee = self.lindhard_splin()(E)
            if E>=15: Ee = self.lindhard_real(E,0.15,14)
        return Ee


    def lindhard_inv(self,E):

        if type(E) is list:
            E = np.array(E)

        cutI, cutII, cutIII, cutIV = self.lindhard(0.04), self.lindhard(0.675), self.lindhard(2.28), self.lindhard(15)
        
        if type(E) is np.ndarray:
            Ee = np.zeros(E.shape)
            ### Region I
            Ee[E<=cutI] = [0] * len([E<=cutI])
            ### Region II
            Ee[(E>cutI) & (E<cutII)] = self.lindhard_low_thres(func="Inv")(E[(E>cutI) & (E<cutII)])
            ### Region III (DAMIC Fit)
            Ee[(E>=cutII) & (E<=cutIII)] = self.lindhard_points(func="Inv")(E[(E>=cutII) & (E<=cutIII)])
            ### Region IV (Seconds interpolation DAMIC Fit)Inv
            Ee[(E>cutIII) & (E<cutIV)] = self.lindhard_splin(func="Inv")(E[(E>cutIII) & (E<cutIV)])
            ### Region V (Lindhard k=0.15)
            Ee[E>=cutIV] = self.lindhard_real_inv(E[E>=cutIV])
        else:
            if E<=cutI: Ee = 0
            if E>cutI and E<cutII: Ee = self.lindhard_low_thres(func="Inv")(E)
            if E>=cutII and E<=cutIII: Ee = self.lindhard_points(func="Inv")(E)
            if E>cutIII and E<cutIV: Ee = self.lindhard_splin(func="Inv")(E)
            if E>=cutIV: Ee = self.lindhard_real_inv(E)
        return Ee


    def lindhard_derivative(self,E):
        
        if type(E) is list:
            E = np.array(E)

        if type(E) is np.ndarray:
            Ee = np.zeros(E.shape)
            ### Region I
            Ee[E<=0.04] = [1e-6] * len([E<=0.04])
            #print(Ee_I)
            ### Region II
            Ee[(E>0.04) & (E<0.675)] = self.lindhard_low_thres(func="Der")(E[(E>0.04) & (E<0.675)])
            ### Region III (DAMIC Fit)
            Ee[(E>=0.675) & (E<=2.28)] = self.lindhard_points(func="Der")(E[(E>=0.675) & (E<=2.28)])
            ### Region IV (Seconds interpolation DAMIC Fit)
            Ee[(E>2.28) & (E<15)] = self.lindhard_splin(func="Der")(E[(E>2.28) & (E<15)])
            ### Region V (Lindhard k=0.15)
            Ee[E>=15] = self.lindhard_real_der(E[E>=15])
        else:
            if E<=0.04: Ee = 1e-6
            if E>0.04 and E<0.675: Ee = self.lindhard_low_thres(func="Der")(E)
            if E>=0.675 and E<=2.28: Ee = self.lindhard_points(func="Der")(E)
            if E>2.28 and E<15: Ee = self.lindhard_splin(func="Der")(E)
            if E>=15: Ee = self.lindhard_real(E,0.15,14)
        return Ee



    """
    def lindhard_inv(self, Ee):
        Computes the numercial inverse function of the lindhard model. This converts ionization energy to nuclear recoils.
        
        return self.interpol_inv(Ee)
    """
    """
    def lindhard_derivative(self, Er):#dx=1e-6):
         Calculates the dEe/dEr useful to some calculus for the differential rate.
        
        if type(Er) is list:
            Er = np.array(Er)
        return self.interpol_der(Er)
    """
