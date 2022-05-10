import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.optimize import bisect 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import UnivariateSpline

from WIMpy import DMUtils as DMU
#from lindhard import *
from lindhard import lindhard_transform 
from differential_rate import *
from simulate_events import *
from likelihood import *


class dm_event(object):
    def __init__(self, N_p_Si, N_n_Si, mass_dm, cross_section, mass_det, t_exp, detection, Eemin=4e-4,Eemax=7, sigma_res_Ee = 4e-4, sigma_Ee = 4e-4, eff = 1, step = 400):
        self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section = N_p_Si, N_n_Si, mass_dm, cross_section
        self.mass_det, self.t_exp = mass_det, t_exp
        self.detection = detection
        self.Eemin, self.Eemax = Eemin, Eemax
        ### FIXME quick estimation of Ermin and Ermax
        l = lindhard_transform(self.Eemin, self.Eemax*3, step)
        self.Ermin, self.Ermax = l.lindhard_inv(self.Eemin), l.lindhard_inv(self.Eemax)
        self.l = lindhard_transform(self.Ermin, self.Ermax, step)
        ### Signal Efficiency resolution
        self.sigma_res_Ee = sigma_res_Ee
        ### Signal Efficiency threshold
        self.sigma_Ee = sigma_Ee
        ### Efficiency
        if type(eff) is str and eff != "efficiency_2020":
            ### If a file name is given interpolates the values
            self.efficiency(eff)
        elif eff == "efficiency_2020":
            self.eff = self.efficiency_2020
        else:
            ### If it is a value uses it as an step function
            self.eff = eff
        self.normalization_signal(step = step)


    def efficiency_2020(self, Ee, plot=False):
        a = 7.5e-2
        b = 75.2
        c = 4.6e-2
        d = -5.62e-3
        e = 9.6e-1
        f = 2.1e-1
        g = 15
        #a,b,c,d,e,f,g = [ 7.54818601e-02,6.60650401e+01,7.19271176e-02,-5.52791252e-03,9.89890621e-01,6.89655309e-01,2.00991160e+01]
        eff = ( 1/( 1+np.exp(-(Ee-a)*b) ) -c )*( (d*Ee+e) + f*np.exp(-g*Ee) ) 
        if plot:
            plt.plot(Ee, eff)
            plt.xlabel(r"$E$ [keV$_{ee}$]")
            plt.ylabel(r"$\varepsilon(E_{ee})$")
        #self.sigma_Ee = 40e-3
        eff[Ee<= self.sigma_Ee] = [0] * len([Ee<=self.sigma_Ee])
        return eff


    def efficiency(self, eff_file, plot=False):
        """ Interpolates the efficiency of the given file.
        """
        df = pd.read_csv(eff_file, comment="#")
        Ee_data, eff_data = df["Ee"].to_numpy(), df["eff"].to_numpy()
        eff_interp = UnivariateSpline(Ee_data, eff_data, k = 2, s = 0)
        def eff_func(Ee):
            """ Function to calculate the efficiency of the given energies
            """
            eff_ = eff_interp(Ee)
            eff_[Ee > Ee_data[-1]] = [eff_data[-1]] * len([Ee > Ee_data[-1]])
            eff_[Ee < Ee_data[0]] = [0] * len([Ee < Ee_data[0]])
            return eff_
        self.eff = eff_func
        self.sigma_Ee, self.sigma_Ee_b = Ee_data[0], Ee_data[0]
        print(self.sigma_Ee, self.sigma_Ee_b)

        if plot:
            Ee = np.linspace(self.Eemin,0.4,400)
            plt.plot(Ee, self.eff(Ee))
            plt.xlabel(r"$E$ [keV$_{ee}$]")
            plt.ylabel(r"$\varepsilon(E_{ee})$")

    def normalization_signal(self, step = 400, n_std = 5):
        """ Calculates the normalization of the signal PDF.
        """

        Enr_space = np.geomspace(self.Ermin, self.Ermax, step)
        if self.detection:
            dRdE = dR_dEe(Enr_space, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.sigma_res_Ee, self.sigma_Ee, self.eff, step, n_std, self.Ermin, self.Ermax)
            self.dRdE_lamb = lambda E: dR_dEe(E, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.sigma_res_Ee, self.sigma_Ee, self.eff, step, n_std, self.Ermin, self.Ermax)

            self.C_sig = simpson(dRdE, self.l.lindhard(Enr_space))
            self.E_space_dist, self.dRdE_dist = self.l.lindhard(Enr_space), dRdE
        else:
            dRdE = DMU.dRdE_standard(Enr_space, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section)
            self.C_sig = simpson(dRdE, Enr_space)
            self.E_space_dist, self.dRdE_dist = Enr_space, dRdE
        #print("Info Events")
        #print(self.C_sig)
        self.s = self.C_sig*self.t_exp*self.mass_det
        #print(self.s)
        self.n_s_det = np.random.poisson(self.s)
        #print(self.n_s_det)


    def simul_ev(self, *bkg_pars, **kwargs):
        """ Simulates n_s_det events for signal.
            bkg_pars contains the dru of bkg and the sensitivity threshold.
        """
        self.sigma_Ee = kwargs["sigma_Ee"] if "sigma_Ee" in kwargs else self.sigma_Ee
        self.sigma_res_Ee = kwargs["sigma_res_Ee"] if "sigma_res_Ee" in kwargs else self.sigma_res_Ee
        step = kwargs["step"] if "step" in kwargs else 400
        Eemin = kwargs["Eemin"] if "Eemin" in kwargs else self.Eemin
        Eemax = kwargs["Eemax"] if "Eemax" in kwargs else self.Eemax

        if "sigma_Ee" in kwargs or "sigma_res_Ee" in kwargs:
            self.normalization_signal(step=step)

        self.signal_ev = random_signal_det(self.n_s_det, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.C_sig, self.detection, Eemin = Eemin, Eemax = Eemax, sigma_res_Ee = self.sigma_res_Ee, sigma_Ee = self.sigma_Ee, eff=self.eff, step = step) 
        #if self.detection:
        #    ### Events in Eee units
        #    self.signal_ev = random_signal_det(self.n_s_det, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.C_sig, self.detection, self.Eemin, self.Eemax, self.sigma_res_Ee, self.sigma_Ee, step) 
        #else:
        #    ### Events in Enr units
        #    self.signal_ev = random_signal_det(self.n_s_det, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.C_sig, self.Eemin, self.Eemax, False)

        if bkg_pars:
            #### Add background events
            dru, E_bkg_cut = bkg_pars[0], bkg_pars[1]
            #self.dru_bkg = dru
            #self.sigma_Ee_b = E_bkg_cut 
            #### Events in Eee
            if not hasattr(self, "bkg_ev"):
                self.simul_bkg(dru, E_bkg_cut)
            self.events = self.signal_ev + self.bkg_ev
            self.signal_generated = True
            print(self.sigma_Ee_b, self.sigma_Ee, self.sigma_res_Ee)
            return self.events
        else:
            self.events = self.signal_ev
            self.signal_generated = True
            return self.signal_ev


    def simul_bkg(self, dru, E_bkg_cut):
        self.dru_bkg = dru
        self.sigma_Ee_b = E_bkg_cut 
        self.C_b = dru*(self.Eemax-E_bkg_cut)
        self.b = self.C_b*self.t_exp*self.mass_det
        self.n_b_det = np.random.poisson(self.b)
        self.sigma_ee_b = E_bkg_cut
        ### Events in Eee
        self.bkg_ev = random_background(self.n_b_det, E_bkg_cut, self.Eemax)
        if not self.detection: self.bkg_ev = self.l.lindhard_inv(self.bkg_ev).tolist()
 

    def likelihood(self, **kwargs):
        bkg = hasattr(self, "bkg_ev")
        ### Cut in signal efficiency
        sigma_Ee = kwargs["sigma_Ee"] if "sigma_Ee" in kwargs else self.sigma_Ee
        sigma_Ee_b = kwargs["sigma_Ee_b"] if "sigma_Ee_b" in kwargs else 8e-4
        sigma_res_Ee = kwargs["sigma_res_Ee"] if "sigma_res_Ee" in kwargs else self.sigma_res_Ee

        if bkg:
            ### Theta = [cross_sec_exponent, n_b]
            bnds = kwargs["bounds"]  if "bounds" in kwargs else  [(-44,-38), (0,1e6)]
            x0 = kwargs["x0"] if "x0" in kwargs else [-40,100]
        else:
            ### Theta = [cross_sec_exponent]
            bnds = kwargs["bounds"]  if "bounds" in kwargs else [(-50,-30)]
            x0 = kwargs["x0"] if "x0" in kwargs else [-40]

        if self.detection:
            theta = minimize(log_likelihood, x0, bounds=bnds,method='L-BFGS-B', args = (self.l.lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, bkg, sigma_Ee, sigma_Ee_b, sigma_res_Ee, self.eff)).x
            print(10**theta)
        else:
            theta = minimize(log_likelihood, x0, bounds=bnds,method='L-BFGS-B', args = (self.events, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, bkg, sigma_Ee, sigma_Ee_b, self.eff)).x
        self.theta = theta

        return theta


    def upper_limit(self, cf = 0.95, plot = False, signal = False, flat_prior = False,**kwargs):
        """ Finds the upper limit for the given DM mass and background.
            Calculates the likelihood for bkg data only and obtains a cross section.
            The upper limit is given by the cross section value that makes dLL = 4.5.
        """
        ### Theta = [cross_sec_exponent, n_b]
        bnds = kwargs["bounds"]  if "bounds" in kwargs else  [[-44,-38], [0,1e6]]
        x0 = kwargs["x0"] if "x0" in kwargs else [-40,100]
        bkg_pars = kwargs["bkg_pars"] if "bkg_pars" in kwargs else [0.1,4e-2]
        step = kwargs["step"] if "step" in kwargs else 200
        if "sigma_Ee" in kwargs: self.sigma_Ee = kwargs["sigma_Ee"]
        if "sigma_res_Ee" in kwargs: self.sigma_res_Ee = kwargs["sigma_res_Ee"]
        #sigma_Ee = kwargs["sigma_Ee"] if "sigma_Ee" in kwargs else self.sigma_Ee
        #sigma_Ee_b = kwargs["sigma_Ee_b"] if "sigma_Ee_b" in kwargs else 4e-4
        #sigma_res_Ee = kwargs["sigma_res_Ee"] if "sigma_res_Ee" in kwargs else self.sigma_res_Ee

        #if not hasattr(self, "bkg_ev"):
        dru, E_bkg_cut = bkg_pars[0], bkg_pars[1]
        self.dru_bkg = dru
        self.sigma_Ee_b = E_bkg_cut

        if "bkg_pars" in kwargs and not signal:
            ### If the background is not yet simulated its calculated.
            self.simul_bkg(dru, E_bkg_cut)
            self.events = self.bkg_ev# if self.detection else self.bkg_ev
            print(len(self.events))
            self.sigma_Ee_b = E_bkg_cut
            Er = self.l.lindhard_inv(self.bkg_ev) if self.detection else self.bkg_ev
        
        elif "bkg_pars" in kwargs and signal:
            ### If the background is not yet simulated its calculated.
            Eemin = kwargs["Eemin"] if "Eemin" in kwargs else self.Eemin
            Eemax = kwargs["Eemax"] if "Eemax" in kwargs else self.Eemax
            if "Eemax" in kwargs:
                l = lindhard_transform(Eemin, kwargs["Eemax"]*3)
                self.l = lindhard_transform(Eemin, l.lindhard_inverse(kwargs["Eemax"]))
            step = kwargs["step"] if "step" in kwargs else 400
            self.simul_ev(*bkg_pars, sigma_Ee=self.sigma_Ee, sigma_res_Ee=self.sigma_res_Ee, Eemin=Eemin, Eemax=Eemax, step=step)
            Er = self.l.lindhard_inv(self.events) # if self.detection else self.events
            
        elif signal and not "bkg_pars" in kwargs:
            ### If the background is not yet simulated its calculated.
            self.simul_ev(*bkg_pars, sigma_Ee=self.sigma_Ee, sigma_res_Ee=self.sigma_res_Ee)
            Er = self.l.lindhard_inv(self.events)# if self.detection else self.events

        elif not signal and not "bkg_pars" in kwargs:
            Er = self.l.lindhard_inv(self.bkg_ev)

        if "sigma_Ee_b" in kwargs: self.sigma_Ee_b = kwargs["sigma_Ee_b"]
        ### Finds the cross section for background only data
        ### This will find the minimum value of lnL (in theory it is the maximum because is -lnL)
        theta_max = minimize(log_likelihood, x0, bounds=bnds,method='L-BFGS-B', args = (Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)).x
        #print("MINIM: ", theta_max)
        self.theta_bkg_only = theta_max
        print(theta_max)
        ### Maximum likelihood value for background only 
        lnL_bkg_only = -log_likelihood(theta_max, Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)

        ### Function to shorten the call of log_likelihood
        lnL_lambda = lambda xs:  -log_likelihood([xs, theta_max[1]], Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)
        #lnL_lambda = lambda theta:  -log_likelihood(theta, Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)

        def prior(xs):
            """ Use a flat prior or a flat prior in log scale
            """
            if flat_prior:
                if bnds[0][0] <= xs <= bnds[0][1]:
                    return 0.0
                else:
                    return -np.inf
            else:
                #return np.log(1/(10**xs*np.log(10**bnds[0][1]/10**bnds[0][0])))
                return -xs*np.log(10)

        ### Calculates the integral of the log-likelihood over the full cross section range
        ### https://stats.stackexchange.com/questions/434145/how-to-integrate-the-marginal-likelihood-numerically
        xs_range_full = np.geomspace(bnds[0][0], bnds[0][1], step)
        ## The maximum of the prior is at the begining of the interval
        lnprior_max = prior(bnds[0][0])
        lnL_full = []
        
        """
        for xs in xs_range_full:
            lnL_i = lambda b:  log_likelihood([xs, b], Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)
            b_i = minimize(lnL_i, [theta_max[1]], bounds=[bnds[1]],method='L-BFGS-B').x[0]
            lnL_full.append( np.exp(lnL_i(b_i) + prior(xs) -lnL_bkg_only -lnprior_max)  ) 
        """

        lnL_full =  np.log(simpson([np.exp(lnL_lambda(xs) + prior(xs) -lnL_bkg_only -lnprior_max) for xs in xs_range_full], 10**xs_range_full ))
        #lnL_full =  np.log(simpson([np.exp(lnL_lambda(xs)-lnL_bkg_only) for xs in xs_range_full], xs_range_full ))

        def lnL_prob(xs_upper):
            """ Function to calculate the cross section at the given confidence level.
                Calculates the xs value in wich integral(L)
            """
            if bnds[0][0] == xs_upper:
                return cf
            elif bnds[0][1] == xs_upper:
                return cf - 1  
            xs_range = np.geomspace(bnds[0][0], xs_upper, step)
            lnL_num = [np.exp(lnL_lambda(xs) + prior(xs)  -lnL_bkg_only -lnprior_max) for xs in xs_range]

            """
            lnL_num = []
            for xs in xs_range:
                lnL_i = lambda b:  log_likelihood([xs, b], Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)
                b_i = minimize(lnL_i, [theta_max[1]], bounds=[bnds[1]],method='L-BFGS-B').x[0]
                print(lnL_bkg_only)
                print(-lnL_i(b_i))
                lnL_num.append( np.exp(-lnL_i(b_i) + prior(xs) - lnL_bkg_only -lnprior_max) ) 
            """

            #lnL_num = [np.exp(lnL_lambda(xs) - lnL_bkg_only ) for xs in xs_range]
            numerator = np.log(simpson(lnL_num, 10**xs_range))
            print("Upper xs: ", xs_upper)
            print("Prob: ", np.exp(numerator - lnL_full))
            return cf - np.exp(numerator - lnL_full)
        self.cross_section_95 = 10**bisect(lnL_prob, bnds[0][0], bnds[0][1], rtol=1/step)
        print(self.cross_section_95)


        dLL = lnL_bkg_only + log_likelihood([np.log10(self.cross_section_95), theta_max[1]], Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)
        print("dLL = {}".format(dLL))
         
        xs_range_full = np.geomspace(bnds[0][0], bnds[0][1], 30)
        if plot:
            plt.plot(xs_range_full, [cf-lnL_prob(xs) for xs in xs_range_full])
            plt.hlines(cf, xs_range_full[0], xs_range_full[-1], color="r")
            plt.axvline(np.log10(self.cross_section_95), color="r")
            plt.xlabel(r"$\sigma$ [cm$^2$]")
            plt.ylabel(r"P($\sigma\leq \sigma_{cf})$")
            #plt.xscale("log")
            plt.show()

        ###Plots the likelihood
        if plot:
            xs_vec = np.linspace(bnds[0][0], bnds[0][1], 30)
            L = []
            for xs in xs_vec:
                theta = [xs, theta_max[1]]
                L.append(log_likelihood(theta, Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff)+lnL_bkg_only)
            plt.axvline(np.log10(self.cross_section_95), color="r")
            plt.plot(xs_vec,L)
            #plt.xscale("log")
            plt.ylabel(r"$ln\mathcal{L}$")
            plt.xlabel(r"$\sigma$ [cm$^{-2}$]")
            plt.ylim(-0.5,4)
            plt.show()

    def upper_limit_dLL(self, deltaLL = 4.5, plot = False, **kwargs):
        """ Finds the upper limit for the given DM mass and background.
            Calculates the likelihood for bkg data only and obtains a cross section.
            The upper limit is given by the cross section value that makes dLL = 4.5.
        """
        ### Theta = [cross_sec_exponent, n_b]
        bnds = kwargs["bounds"]  if "bounds" in kwargs else  [[-44,-38], [0,1e6]]
        x0 = kwargs["x0"] if "x0" in kwargs else [-40,100]
        bkg_pars = kwargs["bkg_pars"] if "bkg_pars" in kwargs else [0.1,4e-4]
        step = kwargs["step"] if "step" in kwargs else 100
        if "sigma_Ee" in kwargs: self.sigma_Ee = kwargs["sigma_Ee"]
        if "sigma_res_Ee" in kwargs: self.sigma_res_Ee = kwargs["sigma_res_Ee"]
        #sigma_Ee = kwargs["sigma_Ee"] if "sigma_Ee" in kwargs else self.sigma_Ee
        #sigma_Ee_b = kwargs["sigma_Ee_b"] if "sigma_Ee_b" in kwargs else 4e-4
        #sigma_res_Ee = kwargs["sigma_res_Ee"] if "sigma_res_Ee" in kwargs else self.sigma_res_Ee

        #if not hasattr(self, "bkg_ev"):
        if bkg_pars:
            ### If the background is not yet simulated its calculated.
            dru, E_bkg_cut = bkg_pars[0], bkg_pars[1]
            self.dru_bkg = dru
            self.simul_bkg(dru, E_bkg_cut)
            self.events = self.bkg_ev
            self.sigma_Ee_b = E_bkg_cut
        Er = self.l.lindhard_inv(self.bkg_ev) if self.detection else self.bkg_ev
        if "sigma_Ee_b" in kwargs: self.sigma_Ee_b = kwargs["sigma_Ee_b"]
        ### Finds the cross section for background only data
        ### This will find the minimum value of lnL (in theory it is the maximum because is -lnL)
        theta_max = minimize(log_likelihood, x0, bounds=bnds,method='L-BFGS-B', args = (Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)).x
        #print("MINIM: ", theta_max)
        self.theta_bkg_only = theta_max

        ### Maximum likelihood value for background only 
        lnL_bkg_only = -log_likelihood(theta_max, Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)

        ### Function to find the roots of lnL-lnL_max+4.5. 
        ### To be able to find the xs that is deltaLL = 4.5 away from the maximum
        #lnL_func = lambda xs_range: -log_likelihood([xs_range, theta_max[1]], Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff) #+lnL_bkg_only+deltaLL
        """
        def lnL_dLL(xs_range):
            #Function to find the value where ln_max- ln_confidence = deltaLL
            
            lnL_func = lambda b: log_likelihood([xs_range, b], Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff) 
            b_min = minimize(lnL_func, x0[1], bounds=[bnds[1]], method='L-BFGS-B').x
            #print(xs_range, b_min)
            return -log_likelihood([xs_range, b_min], Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff)-lnL_bkg_only+deltaLL
        """

        def lnL_dLL(xs_range):
            """ Function to find the value where ln_max- ln_confidence = deltaLL
            """
            return -log_likelihood([xs_range, theta_max[1]], Er, self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, step)-lnL_bkg_only+deltaLL

        self.cross_section_95 = 10**bisect(lnL_dLL, bnds[0][0], bnds[0][1], rtol=1/step)
        print(self.cross_section_95)

        ##Plots the likelihood
        if plot:
            xs_vec = np.linspace(bnds[0][0], bnds[0][1], 100)
            L = []
            for xs in xs_vec:
                theta = [xs, theta_max[1]]
                L.append(lnL_dLL(xs))
            plt.plot(10**xs_vec,L)
            plt.xscale("log")
            #plt.ylim(-lnL_bkg_only-deltaLL,-lnL_bkg_only*1.005)
            #plt.ylim(lnL_lambda(self.cross_section_95),-lnL_bkg_only*1.005)
            plt.ylim(0,deltaLL*1.5)
            plt.ylabel(r"$ln\mathcal{L}$")
            plt.xlabel(r"$\sigma$ [cm$^{-2}$]")
            plt.show()
            #plt.plot(10**xs_vec, [lnL_func(xs) for xs in xs_vec])
            #plt.xscale("log")
            #plt.ylim(0,6)
            #plt.ylabel(r"$\Delta \mathcal{LL}$")
            #plt.xlabel(r"$\sigma$ [cm$^{-2}$]")
            #plt.show()
            #l = np.array([lnL_func(xs) for xs in xs_vec])


    def projected_discovery(self, bkg_pars=None, plot=False, **kwargs):
        """ Given a signal+background data, calculates how can the mass cross section
            and background events can be reconstructed.
        """
        ### Optional arguments
        self.sigma_Ee = kwargs["sigma_Ee"] if "sigma_Ee" in kwargs else self.sigma_Ee
        self.sigma_res_Ee = kwargs["sigma_res_Ee"] if "sigma_res_Ee" in kwargs else self.sigma_res_Ee
        step = kwargs["step"] if "step" in kwargs else 400
        
        ###  If the threshold or resolution changes renormalizes the pdfs
        if "sigma_Ee" in kwargs or "sigma_res_Ee" in kwargs:
            self.normalization_signal()
            self.signal_generated = False

        ### If the signal is not yet generated. Generates it
        if not hasattr(self, "signal_generated"):
            if bkg_pars:
                self.sigma_Ee_b = bkg_pars[1]
                self.simul_ev(*bkg_pars)
            else:
                self.simul_ev()

        ### The likelihood bounds and initial parameters can be given
        bnds = kwargs["bounds"]  if "bounds" in kwargs else  [[-44,-38],[0,30],[0,1e6]]
        x0 = kwargs["x0"] if "x0" in kwargs else [-40,5,100]

        ###Plots the likelihood agains the DM mass
        if plot:
            lnL_func = lambda mx_range: -log_likelihood([np.log10(self.cross_section), mx_range, self.n_b_det], self.l.lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff) +lnL_bkg_only+deltaLL
            mx_vec = np.linspace(bnds[1][0], bnds[1][1], 100)
            L = []
            for m in mx_vec:
                theta = [np.log10(self.cross_section), m,self.n_b_det]
                L.append(-log_likelihood(theta, self.l.lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff))
            plt.plot(mx_vec,L)
            plt.xscale("log")
            plt.show()

        ### Calculates the likelihood with or without background.
        ### Calculates the confidence intervals of the theta parameters
        if bkg_pars:
            theta_max = minimize(log_likelihood, x0, bounds=bnds,method='L-BFGS-B', args = (self.l.lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff)).x

            theta_errors = theta_confidence(theta_max, self.l.lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, True, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, bnds)
        else:
            theta_max = minimize(log_likelihood, x0, bounds=bnds,method='L-BFGS-B', args = (self.l.lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, False, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff)).x
            theta_errors = theta_confidence(theta_max, self.l.lindhard_inv(self.events), self.Ermin, self.Ermax, self.N_p_Si, self.N_n_Si, self.mass_dm, self.mass_det*self.t_exp, self.detection, False, self.sigma_Ee, self.sigma_Ee_b, self.sigma_res_Ee, self.eff, bnds)
        
        self.theta = theta_max
        self.theta_confidence = theta_errors



    def verbose(self):
        print("Number of Signal events: ", self.n_s_det)
        if hasattr(self, "n_b_det"): print("Number of Background events: ", self.n_b_det)
        if hasattr(self, "theta"): print("Likelihood parameters: ", self.theta)
        if hasattr(self, "b_fit"): print("Background Only-Likelihood Events : ", self.b_fit)
        if hasattr(self, "dLL"): print("dLL: ", self.dLL)
        if hasattr(self, "cross_section_95"): print("Upper Limit Cross Section: ", self.cross_section_95)
        if hasattr(self, "theta_confidence"): print("Theta Confidence intervals: ", self.confidence)

    def dRdE(self, step = 400, Eemin=4e-3, Eemax=7, n_std = 5):
        E_space = np.geomspace(self.l.lindhard_inv(Eemin), self.l.lindhard_inv(Eemax), step) # Nuclear recoils
        if self.detection:
            dRdE = dR_dEe(E_space, self.N_p_Si, self.N_n_Si, self.mass_dm, self.cross_section, self.sigma_res_Ee, self.sigma_Ee, self.eff, step, n_std, self.Ermin, self.Ermax)
            E_space = self.l.lindhard(E_space) # Return in Eee units
            #Compare theoretical and detection if sigma_res_Ee is 0 
            #dRdE = dRdE/self.l.lindhard_derivative(E_space)
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

        label = kwargs["label"] if "label" in kwargs else None
        n_std = kwargs["n_std"] if "n_std" in kwargs else 5

        if var_name == "dRdE":
            E_space, dRdE_space = self.dRdE(step, Eemin=Eemin, Eemax=Eemax, n_std=n_std)
            if not self.detection:
                ### Compare theoretical dRdE in Eee
                dRdE_space = dRdE_space/self.l.lindhard_derivative(E_space)
                E_space = self.l.lindhard(E_space)
                print("Number of theory signal Eee: ", self.mass_det*self.t_exp*simpson(dRdE_space,E_space))
            plt.plot(E_space, dRdE_space, label=label)
            if self.detection:
                plt.xlabel(r"$E_{ee}$ [keV$_{ee}$]")
                plt.ylabel(r"$dR/dE_{ee}$ [events/keV$_{ee}\cdot$kg$\cdot$days]")
            else:
                plt.xlabel(r"$E_{NR}$ [keV]")
                plt.ylabel(r"$dR/dE_{NR}$ [events/keV$\cdot$kg$\cdot$days]")

        elif var_name == "fs":
            E_space, dRdE_space = self.dRdE(step, Eemin=Eemin, Eemax=Eemax, n_std=n_std)
            if not self.detection:
                ### Cross check dRdE theoretical in Eee
                dRdE_space = dRdE_space/self.l.lindhard_derivative(E_space)
                E_space = self.l.lindhard(E_space)
            plt.plot(E_space, dRdE_space/self.C_sig, label=label)
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
                fb = 1/(self.Eemax-self.Eemin)*self.l.lindhard_derivative(E_space)
                plt.xlabel(r"$E_{NR}$ [keV]")
            plt.plot(E_space, fb)

        elif var_name == "fsb":
            E_space, dRdE_space = self.dRdE(step, Eemin=Eemin, Eemax=Eemax, n_std=n_std)
            if self.detection:
                fb = np.array([self.dru_bkg] * step)
                fb[E_space<self.sigma_Ee_b]  = [0] * len(fb[E_space<self.sigma_Ee_b])
                plt.xlabel(r"$E_{ee}$ [keV$_{ee}$]")
            else:
                fb = self.dru_bkg*self.l.lindhard_derivative(E_space)
                plt.xlabel(r"$E_{NR}$ [keV]")
            fsb = (dRdE_space+fb)*self.t_exp*self.mass_det/len(self.events)#(self.n_s_det+self.n_b_det)
            plt.plot(E_space, fsb, label=label)
            plt.ylabel(r"$f_{s+b}(E|M)$")

