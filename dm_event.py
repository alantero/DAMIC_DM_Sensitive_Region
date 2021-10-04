import numpy as np
from scipy.integrate import simpson
from pynverse import inversefunc
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from WIMpy import DMUtils as DMU
from lindhard import *
from differential_rate import *
from simulate_events import *
from likelihood import *

class dm_event(object):
    def __init__(self, N_p_Si, N_n_Si, mass_dm, cross_section, mass_det, t_exp, detection, Eemin=4e-3,Eemax=7):
        self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section = N_p_Si, N_n_Si, mass_dm, cross_section
        self.mass_det, self.t_exp = mass_det, t_exp
        self.detection = detection
        self.Eemin, self.Eemax = Eemin, Eemax
        self.Ermin, self.Ermax = lindhard_inv(self.Eemin), lindhard_inv(self.Eemax)
        self.C_sig = self.normalization_signal()
        self.s = self.C_sig*self.t_exp*self.mass_det
        self.n_s_det = np.random.poisson(self.s)


    def normalization_signal(self, step = 400):
        """ Calculates the normalization of the signal PDF.
        """

        Enr_space = np.geomspace(self.Ermin, self.Ermax, step)
        if self.detection:
            dRdE = dR_dEe(Enr_space, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section)
            C_sig = simpson(dRdE, lindhard(Enr_space))
        else:
            dRdE = DMU.dRdE_standard(Enr_space, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section)
            C_sig = simpson(dRdE, Enr_space)
        return C_sig



    def simul_ev(self, *bkg_pars):
        """ Simulates n_s_det events for signal.
            bkg_pars contains the dru of bkg and the sensitivity threshold.
        """
        if self.detection:
            ### Events in Eee units
            self.signal_ev = random_signal_det(self.n_s_det, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.C_sig, self.Eemin, self.Eemax) 
        else:
            ### Events in Enr units
            self.signal_ev = random_signal(self.n_s_det, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.C_sig, self.Eemin, self.Eemax, False)

        if bkg_pars:
            ### Add background events
            dru, E_bkg_cut = bkg_pars[0], bkg_pars[1]
            self.C_b = dru*(self.Eemax-E_bkg_cut)
            self.b = self.C_b*self.t_exp*self.mass_det
            self.n_b_det = np.random.poisson(self.b)
            ### Events in Eee
            self.bkg_ev = random_background(self.n_b_det, E_bkg_cut, self.Eemax)
            if not self.detection: self.bkg_ev = lindhard_inv(self.bkg_ev).tolist()
            self.events = self.signal_ev + self.bkg_ev
            return self.events
        else:
            self.events = self.signal_ev
            return self.signal_ev



    def likelihood(self, deltaLL = False, errors = False, **kwargs):
        bkg = hasattr(self, "bkg_ev")
        ### Cut in signal efficiency
        sigmaEe = kwargs["sigmaEe"] if "sigmaEe" in kwargs else 4e-3
        sigmaEe_b = kwargs["sigmaEe_b"] if "sigmaEe_b" in kwargs else 8e-3

        if bkg:
            ### Theta = [cross_sec_exponent, n_b]
            bnds = kwargs["bounds"]  if "bounds" in kwargs else  [(-44,-38), (0,1e6)]
            x0 = kwargs["x0"] if "x0" in kwargs else [-40,100]

        else:
            ### Theta = [cross_sec_exponent]
            bnds = kwargs["bounds"]  if "bounds" in kwargs else [(-50,-30)]
            x0 = kwargs["x0"] if "x0" in kwargs else [-40]

        if self.detection:
            theta = minimize(log_likelihood, x0, bounds=bnds,method='L-BFGS-B', args = (lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, bkg)).x
        else:
            theta = minimize(log_likelihood, x0, bounds=bnds,method='L-BFGS-B', args = (self.events, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, bkg)).x
        self.theta = theta

        if deltaLL:
            ### Likelihood value for the fitted parameters s+b
            lnL_sb = log_likelihood(self.theta, lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, bkg) 
            ### Background only likelihood
            b_fit = minimize(log_likelihood_bkg, x0[1], bounds=[bnds[1]], method='L-BFGS-B', args = (lindhard_inv(self.events), self.Ermin, self.Ermax, sigmaEe_b)).x
            lnL_b = log_likelihood_bkg(b_fit, self.events, self.Eemin, self.Eemax, sigmaEe_b )
            #print("b: ", lnL_b)
            #print("sb: ", lnL_sb)
            self.dLL = lnL_b-lnL_sb
            #print("DeltaLL: ", self.dLL)
            self.b_fit = b_fit
            ### To find the minimum sensitivity measurable with the minimizer
            dLL_thres = kwargs["dLL_thres"] if "dLL_thres" in kwargs else 4.5
            return (self.dLL - dLL_thres)**2

        if errors:
            theta_errors = theta_confidence(theta, self.events, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, bkg)
        return theta



    def likelihood_sigma_i(self, sigma_0, *bkg_pars):
        self.cross_section = 10**sigma_0
        self.simul_ev(*bkg_pars)
        dLL = self.likelihood(deltaLL = True)
        return dLL



    def minimize_sensitivity(self, sigma_0 = -42, dLL_thres = 4.5, bnds = [[-43,-41]], *bkg_pars):
        """ For the given DM mass calculates what is the minimum cross section
            that gives dLL = 4.5. Meaning that is discoverable.
        """
        self.original_cross_section = self.cross_section
        minimum_cross_section = minimize(self.likelihood_sigma_i, sigma_0 ,bounds=bnds, args = (bkg_pars) ).x
        self.cross_section = self.original_cross_section



    def verbose(self):
        print("Number of Signal events: ", self.n_s_det)
        if hasattr(self, "n_b_det"): print("Number of Background events: ", self.n_b_det)
        if hasattr(self, "theta"): print("Likelihood parameters: ", self.theta)
        if hasattr(self, "b_fit"): print("Background Only-Likelihood Events : ", self.b_fit)
        if hasattr(self, "dLL"): print("dLL: ", self.dLL)



    def dRdE(self, step = 400, Eemin=4e-3, Eemax=7):
        E_space = np.geomspace(lindhard_inv(Eemin), lindhard_inv(Eemax), step) # Nuclear recoils
        if self.detection:
            dRdE = dR_dEe(E_space, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section)
            E_space = lindhard(E_space) # Return in Eee units
        else:
            dRdE = DMU.dRdE_standard(E_space, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section)
        return E_space, dRdE



    def plot_var(self,var_name, step = 100, **kwargs):
        """ Make different plots of the variables. 
        """
        if "Eemin" in kwargs:
            Eemin = kwargs["Eemin"]
        else:
            Eemin = self.Eemin
        if "Eemax" in kwargs:
            Eemax = kwargs["Eemax"]
        else:
            Eemax = self.Eemax

        if var_name == "dRdE":
            E_space, dRdE_space = self.dRdE(step, Eemin=Eemin, Eemax=Eemax)
            plt.plot(E_space, dRdE_space)

            if self.detection:
                plt.xlabel(r"$E_{ee}$ [keV$_{ee}$]")
                plt.ylabel(r"$dR/dE_{ee}$ [events/keV_{ee}$\cdot$kg$\cdot$days]")
            else:
                plt.xlabel(r"$E_{NR}$ [keV]")
                plt.ylabel(r"$dR/dE_{NR}$ [events/keV$\cdot$kg$\cdot$days]")

        elif var_name == "fs":
            E_space, dRdE_space = self.dRdE(step, Eemin=Eemin, Eemax=Eemax)
            plt.plot(E_space, dRdE_space/self.C_sig)
            if self.detection:
                plt.xlabel(r"$E_{ee}$ [keV$_{ee}$]")
            else:
                plt.xlabel(r"$E_{NR}$ [keV]")
            plt.ylabel(r"$f_s(E|M)$")

        elif var_name == "fb":
            if self.detection:
                E_space = np.geomspace(Eemin, Eemax, step)
                fb = [1/(self.Eemax-self.Eemin)] * step
                plt.xlabel(r"$E_{ee}$ [keV$_{ee}$]")
            else:
                E_space = np.geomspace(self.Ermin, self.Ermax, step)
                fb = 1/(self.Eemax-self.Eemin)*lindhard_derivative(E_space)
                plt.xlabel(r"$E_{NR}$ [keV]")
            plt.plot(E_space, fb)

        elif var_name == "fsb":
            E_space, dRdE_space = self.dRdE(step, Eemin=Eemin, Eemax=Eemax)
            if self.detection:
                fb = np.array([1/(self.Eemax-self.Eemin)] * step)
                plt.xlabel(r"$E_{ee}$ [keV$_{ee}$]")
            else:
                E_space = np.geomspace(self.Ermin, self.Ermax, step)
                fb = 1/(self.Eemax-self.Eemin)*lindhard_derivative(E_space)
                plt.xlabel(r"$E_{NR}$ [keV]")
            #fsb = (dRdE_space*self.n_s_det/self.C_sig+fb*self.n_b_det)/(self.n_s_det+self.n_b_det)
            fsb = (dRdE_space/self.C_sig+fb)
            plt.plot(E_space, fsb)
            plt.ylabel(r"$f_{s+b}(E|M)$")

        elif var_name == "cdf":
            E_space = np.geomspace(self.Ermin, self.Ermax, step)
            cdf = [cdf_signal(E, N_p_Si, N_n_Si, mass_dm, cross_section, C_sig) for E in E_space]
            if self.detection:
                plt.plot(lindhard(E_space), cdf)
                plt.xlabel(r"$E_{ee}$ [keV$_{ee}$]")
            else:
                plt.plot(E_space, cdf)
                plt.xlabel(r"$E_{NR}$ [keV]")
            plt.ylabel(r"$F_s(E|M)$")
            plt.xscale("log")

        elif var_name == "inv_cdf":
            u = np.linspace(1e-1,1,step)
            inv_l = []
            bnds = [ [lindhard_inv(self.Eemin),lindhard_inv(self.Eemax)] ]
            def inv(x,y0):
                return (cdf_signal(x,self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.C_sig, self.detection)-y0)**2
            for x in u:
                inv_l.append(minimize(inv, 0.1, args = (x), method="L-BFGS-B", tol=1e-30).x[0])
            if self.detection:
                plt.plot(u,lindhard(inv_l))
                plt.ylabel(r"$F_s^{-1}(E|M)$ [keV_ee}]")
            else:
                plt.plot(u,inv_l)
                plt.ylabel(r"$F_s^{-1}(E|M)$ [keV]")
            plt.xlabel(r"$F_s(E|M)$")
