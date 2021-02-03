""" 
This module simulates GNSS observations.

"""
import geopandas as gpd

import pandas as pd
import numpy as np
from scipy.stats import rice, norm
from simulator.gnssData import GNSSData
from math import pi

def simulate(observations:gpd.GeoDataFrame,map:) -> gpd.GeoDataFrame:




def observe(points,map,SSLB, mu_,msr_noise):
    """Simulates a set of satellite observations
    Parameters
    ----------
    map: map of area

    points : ReceiverPoints
        position of receiver

    msr_noise : float
        Variance for the simulated signal strengths
    
    Returns
    -------
    observations : Observations
        GNSS readings from receiver 
    """
    #first get the satellite locations with blank signal readings
    observations=locate_satellites(points,map)
    bounded = bound_elevations(observations) 
    bounded.loc[:,'fresnel']= map.fresnel(bounded)
    #model the signal
    bounded.loc[:,'ss'], bounded.loc[:,'pr'] = model_signal(bounded, SSLB,mu_,msr_noise)
    return bounded

def model_signal(observations, SSLB=10, mu_=35,msr_noise=5):
    """
    Parameters
    ----------
    observations : Observations
        satellite  and receiver locations

    SSLB : float
        lowest signal strength that returns a reading

    msr_noise : float
        Variance for the simulated signal strengths
    
    Returns
    -------
    ss : np.float64
        signal strength of observations
    pr : np.float64
        psuedorange of observations
    
    Notes on implementation:
    - Using the models described in Irish et. al. 2014 - Belief Propagation Based Localization and Mapping Using Sparsely Sampled GNSS SNR Measurements
    - NLOS rss follows a lognormal distribution (i.e. normally distributed on a dB scale) with parameters given as mu = 18dB lower than reference power (see below) and sigma = 10dB
    - LOS rss follows a Rician distirbution with K of 2, indicating mdoerate fading conditions and refernce power Omega as max of all readings, which we simulate here as 45dB.
    - from the above we obtain the Rician parameters of nu^2 =K/(1+K)*Omega and sigma^2=Omega/(2*(1+K))
    - N.B for the scipy.stats implementation we have scale = sigma and shape b = nu/sigma
    - Rician needs converting to a dB scale
    """
    # Omega_dB = 45 #reference power which we may tweak
    # Omega = 10**(Omega_dB/20) 
    # K= 2
    # nu = (2/(1+K) * Omega)**0.5
    # sigma = (Omega / (2 * (1+K)))**0.5
    # b=nu/sigma
    
    # rice_=rice(b,scale=sigma)
    # is_los_ = self.map.is_los(observations)
    # ss_ = np.where(is_los_,rice.rvs(2, scale=10, size=len(is_los_)), lognorm.rvs(18, scale=10, size=len(is_los_)))
    
    mus = mu_ - observations.fresnel
    sigmas=np.ones_like(observations.ss)*msr_noise
    ss_ =norm.rvs(mus,sigmas)
    
    ss = np.where(ss_ > SSLB, ss_, np.nan)
    pr = np.zeros_like(observations.ss)

    return ss, pr


