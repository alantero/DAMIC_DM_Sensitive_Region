import numpy as np
from scipy.integrate import simpson
import emcee

from WIMpy import DMUtils as DMU
from lindhard import *
from differential_rate import *
import corner


def log_likelihood_bkg(b, Ee, Eemin, Eemax, sigma_Ee_b = 8e-3):
    """ Calculates the log-likelihood for the background only hypothesis.
        It needs the energy events in ionization energy.
        It also needs the energy interval of the fiducial region in ionization energy.
        E_cut is the cut in the detection efficiency.
    """
    ### To be able to run list type
    if type(Ee) is list:
        Ee = np.array(Ee)

    ### If there is no event bellow the efficiency cut
    ### The log-likelihood wont be divergent.
    if len(Ee[Ee>sigma_Ee_b]) == len(Ee):
        ### Fiducial region length
        dEee = Eemax-Eemin
        ### Uniform distribution for background
        fb_eff = len(Ee)*np.log(b/dEee)
        ### Returns the negative log-likelihood
        _lnL = b - fb_eff
        return _lnL
    else:
        _lnL = -np.inf
    return _lnL



def log_likelihood(theta,Er,Ermin,Ermax, N_p_Si, N_n_Si, m_x,texp_mass,detection=False, background=False, **kwargs):
    """ Calculates the log likelihood of the given Events signals Er.
        Er must be in Nuclear recoil units.
        The loglikelihood can be the theoretical (returning in nuclear recoil units)
        or the detected (returning in Eee units).
        Background can also be added.
    """
    
    ### Converts list into array
    if type(Er) is list:
        Er = np.array(Er)

    ### Reconstruction efficiency thresholds
    ## Signal
    if "sigma_Ee" in kwargs:
        sigma_Ee = kwargs["sigma_Ee"]
    else:
        sigma_Ee = 4e-3
    ## Background
    if "sigma_Ee_b" in kwargs:
        sigma_Ee_b = kwargs["sigma_Ee_b"]
    else:
        sigma_Ee_b = 8e-3

    ### Fitting parameters
    ## cross section is equal to 10 to sigma_exp
    if background:
        sigma_exp, b = theta
    else:
        sigma_exp = theta
        
    ### Cross section real value
    sigma_p = np.power(10, sigma_exp)
    
    ### Fiducial region
    Enr_space = np.geomspace(Ermin,Ermax,100)
    
    if detection:
        ### Detected differential rate (Ionization energy) for the given energy values
        dR = dR_dEe(Er, N_p_Si, N_n_Si, m_x, sigma_p)
        ### Differential rate over the fiducial region
        dR_space = dR_dEe(Enr_space, N_p_Si, N_n_Si, m_x, sigma_p, sigma_Ee = sigma_Ee)
        ### Integral of the differential rate over the fidutial region
        C_sigma0 = simpson(dR_space,lindhard(Enr_space))
    else:
        ### Theoretical differential rate (Nuclear recoils) for the given energy values
        dR = DMU.dRdE_standard(Er, N_p_Si, N_n_Si, m_x, sigma_p)
        ### Differential rate over the fiducial region
        dR_space = DMU.dRdE_standard(Enr_space, N_p_Si, N_n_Si, m_x, sigma_p) 
        ### Integral of the differential rate over the fidutial region
        C_sigma0 = simpson(dR_space,Enr_space)
        
        
    ### Normalized PDF signal and expected signal events
    fs = np.zeros([len(Er)])
    ## Reconstruction efficiency cut
    fs[Er>lindhard_inv(sigma_Ee)] = dR[Er>lindhard_inv(sigma_Ee)]/C_sigma0
    ## Expected signal events
    s = C_sigma0*texp_mass

    if background:
        ### Adds the background to the likelihood
        dEee = lindhard(Ermax)-lindhard(Ermin)
        ## Calculates the number of events above the efficiency threshold
        # En teoria deberian ser todos si esta bien simulado¿
        Er_eff = np.array(Er)[np.array(Er)>lindhard_inv(sigma_Ee_b)]
        fb = np.zeros([len(Er)])
        if detection:
            fb[Er>lindhard_inv(sigma_Ee_b)] = [1/dEee] * len(Er_eff)
        else:
            fb[Er>lindhard_inv(sigma_Ee_b)] = 1/dEee*lindhard_derivative(Er_eff)
        ### log likelihood
        _lnL = (s+b) - np.sum( np.log(s*fs + b*fb) )
    else:
        ### Signal only log likelihood
        _lnL = s - np.sum(np.log(s*fs))        

    return _lnL


def log_prior(theta, background):
    """ Defines the limits of the fitting parameters.
    """
    if background:
        ### Theta = [sigma, b]
        if -44 < theta[0] < 38 and 0 < theta[1] < 1e6:
            return 0.0
        return -np.inf       
    else:
        ### Theta = sigma
        if -44 < theta < 38:
            return 0.0
        return -np.inf       

def log_probability(theta, Er, Ermin, Ermax, N_p_Si, N_n_Si, m_x, texp_mass, detection=False, background=False):
    lp = log_prior(theta, background)
    if not np.isfinite(lp):
        return -np.inf
    return lp - log_likelihood(theta, Er, Ermin,Ermax,N_p_Si, N_n_Si, m_x, texp_mass, detection, background)

def theta_confidence(theta, Er, Ermin, Ermax, N_p_Si, N_n_Si, m_x, texp_mass, detection=False, background=False, **kwargs):
    if background:
        pos = theta + 1e-4 * np.random.randn(32, len(theta))
    else:
        pos = theta + 1e-4 * np.random.randn(32, 1)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(Er,Ermin,Ermax,N_p_Si,N_n_Si,m_x,texp_mass,detection,background))
    sampler.run_mcmc(pos, 5000, progress=True);

    samples = sampler.get_chain()

    tau = sampler.get_autocorr_time() 
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    inds = np.random.randint(len(flat_samples), size=100)

    theta_err = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [5, 50, 95])
        q = np.diff(mcmc)
        #txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        #txt = txt.format(mcmc[1], q[0], q[1], "\sigma")
        #display(Math(txt))
        if i == 0:
            print("95th percentil: ", np.power(10,mcmc[1]+q[1]))
            print("Central vlue: ", np.power(10,mcmc[1]))
            print("5th percentile: ", np.power(10,mcmc[1]-q[0]))
        else:
            print("95th percentil: ", mcmc[1]+q[1])
            print("Central vlue: ", mcmc[1])
            print("5th percentile: ", mcmc[1]-q[0])

        theta_err.append([mcmc[1]-q[0],mcmc[1],mcmc[1]+q[1]])
    return theta_err

