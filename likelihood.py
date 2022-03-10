import numpy as np
from scipy.integrate import simpson
import emcee
import corner

from WIMpy import DMUtils as DMU
#from lindhard import *
from lindhard import lindhard_transform
from differential_rate import *


def log_likelihood_bkg(b, Ee, Eemin, Eemax, sigma_Ee_b = 8e-4):
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


def log_likelihood(theta,Er,Ermin,Ermax, N_p_Si, N_n_Si, m_x,texp_mass,detection=False, background=False, sigma_Ee = 4e-4, sigma_Ee_b = 4e-4, sigma_res_Ee = 4e-4, eff = 1, step=200,**kwargs):
    """ Calculates the log likelihood of the given Events signals Er.
        Er must be in Nuclear recoil units.
        The loglikelihood can be the theoretical (returning in nuclear recoil units)
        or the detected (returning in Eee units).
        Background can also be added.
    """

    ### Converts list into array
    if type(Er) is list:
        Er = np.array(Er)

    l = lindhard_transform(Ermin, Ermax, step)

    ### Fitting parameters
    ## cross section is equal to 10 to sigma_exp
    if len(theta) < 3:
        if background:
            sigma_exp, b = theta
        else:
            sigma_exp = theta
    elif len(theta) == 3:
        ### m_x is still needed as input parameter for simplicity
        ### But the value is rewrited with the one in theta
        sigma_exp, m_x, b = theta

    ### Cross section real value
    sigma_p = np.power(10, sigma_exp)
    
    ### Fiducial region
    Enr_space = np.geomspace(Ermin,Ermax,step)
    
    if detection:
        ### Detected differential rate (Ionization energy) for the given energy values
        dR = dR_dEe(Er, N_p_Si, N_n_Si, m_x, sigma_p, sigma_res_Ee, sigma_Ee, eff, Ermin=Ermin, Ermax=Ermax)
        ### Differential rate over the fiducial region
        dR_space = dR_dEe(Enr_space, N_p_Si, N_n_Si, m_x, sigma_p, sigma_res_Ee, sigma_Ee, eff, Ermin=Ermin, Ermax=Ermax)
        ### Integral of the differential rate over the fidutial region
        C_sigma0 = simpson(dR_space,l.lindhard(Enr_space))
    else:
        ### Theoretical differential rate (Nuclear recoils) for the given energy values
        dR = DMU.dRdE_standard(Er, N_p_Si, N_n_Si, m_x, sigma_p)
        ### Differential rate over the fiducial region
        dR_space = DMU.dRdE_standard(Enr_space, N_p_Si, N_n_Si, m_x, sigma_p) 
        ### Integral of the differential rate over the fidutial region
        C_sigma0 = simpson(dR_space,Enr_space)
    
    if detection:
        ### Normalized PDF signal and expected signal events
        fs = np.zeros([len(Er)])
        ## Reconstruction efficiency cut
        fs[Er>l.lindhard_inv(sigma_Ee)] = dR[Er>l.lindhard_inv(sigma_Ee)]/C_sigma0
    else:
        fs = dR/C_sigma0
    ## Expected signal events
    s = C_sigma0*texp_mass

    if background:
        ### Adds the background to the likelihood
        dEee = l.lindhard(Ermax)-l.lindhard(Ermin)
        ## Calculates the number of events above the efficiency threshold
        Er_eff = np.array(Er)[np.array(Er)>l.lindhard_inv(sigma_Ee_b)]
        fb = np.zeros([len(Er)])
        if detection:
            fb[Er>l.lindhard_inv(sigma_Ee_b)] = [1/dEee] * len(Er_eff)
        else:
            fb = l.lindhard_derivative(Er)/dEee
        ### log likelihood
        #print("fs: ", fs)
        #print("fb: ", fb)
        #print("s:", s)
        #print("C_sigma0: ", C_sigma0)
        #print("b: ", b)
        #print("Sum: ", np.log(s*fs + b*fb))
        _lnL = (s+b) - np.sum( np.log(s*fs + b*fb) )
        #print(_lnL)
    else:
        ### Signal only log likelihood
        _lnL = s - np.sum(np.log(s*fs))        
    return _lnL


def log_prior(theta, background, bounds = None):
    """ Defines the limits of the fitting parameters.
    """
    if len(theta) < 3:
        if background:
            ### Theta = [sigma, b]
            if not bounds:
                if -np.inf < theta[0] < 0 and 0 < theta[1] < 1e6:
                    return 0.0
                return -np.inf      
            else:
                if bounds[0][0] < theta[0] < bounds[0][1] and bounds[1][0] < theta[1] < bounds[1][1]:
                    return 0.0
                return -np.inf      
        else:
            ### Theta = sigma
            if not bounds:
                if -44 < theta < 38:
                    return 0.0
                return -np.inf       
            else:
                if bounds[0][0] < theta < bounds[0][1]:
                    return 0.0
                return -np.inf       
    elif len(theta) == 3:
        if not bounds:
            if -44 < theta[0] < -41 and 0.3 < theta[1] < 15 and 0 < theta[2] < 1e6:
                return 0.0
            return -np.inf       
        else:
            if bounds[0][0] < theta[0] < bounds[0][1] and bounds[1][0] < theta[1] < bounds[1][1] and bounds[2][0] < theta[2] < bounds[2][1]:
                return 0.0
            return -np.inf      
 


def log_probability(theta, Er, Ermin, Ermax, N_p_Si, N_n_Si, m_x, texp_mass, detection=False, background=False, sigma_Ee=4e-4, sigma_Ee_b=4e-4, sigma_res_Ee=4e-4, eff=1, bounds=None):

    lp = log_prior(theta, background, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp - log_likelihood(theta, Er, Ermin,Ermax,N_p_Si, N_n_Si, m_x, texp_mass, detection, background, sigma_Ee, sigma_Ee_b, sigma_res_Ee, eff) 

def theta_confidence(theta, Er, Ermin, Ermax, N_p_Si, N_n_Si, m_x, texp_mass, detection=False, background=False, sigma_Ee=4e-4, sigma_Ee_b=4e-4, sigma_res_Ee=4e-4, eff=1, bounds=None, plot=True, **kwargs):
    """ Calculates the confidence interval of the theta parameters using the emcee method.
        https://emcee.readthedocs.io/en/stable/
        Can plot the correlation regions between parameters.
    """

    if len(theta) < 3:
        ### Theta can contain cross section and background
        if background:
            pos = theta + 1e-4 * np.random.randn(32, len(theta))
            labels = [r"$\sigma$",r"$b$"]
        else:
            pos = theta + 1e-4 * np.random.randn(32, 1)
            labels = [r"$\sigma$"]
    elif len(theta) == 3:
        ### Theta contains the dark matter mass 
        pos = theta + 1e-4 * np.random.randn(32, len(theta))
        labels = [r"$\sigma$",r"$m_{\chi}$",r"$b$"]

    ### emcee calculations
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(Er,Ermin,Ermax,N_p_Si,N_n_Si,m_x,texp_mass,detection,background,sigma_Ee,sigma_Ee_b,sigma_res_Ee,eff,bounds))
    sampler.run_mcmc(pos, 5000, progress=True)

    samples = sampler.get_chain()

    tau = sampler.get_autocorr_time() 
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    inds = np.random.randint(len(flat_samples), size=100)

    if plot:
        fig = corner.corner(flat_samples, labels=labels, truths=theta)

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
            theta_err.append(np.power(10,[mcmc[1]-q[0],mcmc[1],mcmc[1]+q[1]]).tolist())
        else:
            print("95th percentil: ", mcmc[1]+q[1])
            print("Central vlue: ", mcmc[1])
            print("5th percentile: ", mcmc[1]-q[0])
            theta_err.append([mcmc[1]-q[0],mcmc[1],mcmc[1]+q[1]])
    return theta_err

