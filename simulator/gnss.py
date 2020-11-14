import pandas as pd
import numpy as np
from scipy.stats import rice, norm
from simulator.gnss_data import GNSSData
from math import pi

""" 
=========================================
GNSS Emulator for GNSS Map data collection 
=========================================

This module simulates GNSS observations of signal strength 

# include the fresnel value for resampling strengths

"""

# helper function to construct dataframe with correct structure
def Observations(x=None,y=None,z=None,t=None,svid=None,sv_x=None,sv_y=None,sv_z=None,fresnel=None,ss=None,pr=None):
    #change length of empty arrays to non-zero if needed 
    params=[x,y,z,t,svid,sv_x,sv_y,sv_z,fresnel,ss,pr]
    n= max([0]+ [len(p) for p in params if p is not None])
    zero_params=[
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype='datetime64[s]'),
        np.zeros( shape=(n, ) ,dtype=np.str),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64)
    ]

    params=[p if p is not None else q for p,q in zip(params,zero_params)]
    
    return pd.DataFrame({
            'x':params[0],
            'y':params[1],
            'z':params[2],
            't':params[3],
            'svid':params[4],
            'sv_x':params[5],
            'sv_y':params[6],
            'sv_z':params[7],
            'fresnel':params[8],
            'ss':params[9],
            'pr':params[10]
            })

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
    observations=locateSatellites(points,map)
    bounded = boundElevations(observations) 
    bounded['fresnel']= map.fresnel(bounded)

    #model the signal
    observations.ss, observations.pr = modelSignal(observations, SSLB,mu_,msr_noise)

    return observations

def modelSignal(observations, SSLB=10, mu_=35,msr_noise=5):
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
    # isLos_ = self.map.isLos(observations)
    # ss_ = np.where(isLos_,rice.rvs(2, scale=10, size=len(isLos_)), lognorm.rvs(18, scale=10, size=len(isLos_)))
    
    mus = mu_ - observations.fresnel
    sigmas=np.ones_like(observations.ss)*msr_noise
    ss_ =norm.rvs(mus,sigmas)
    
    ss = np.where(ss_ > SSLB, ss_, np.nan)
    pr = np.zeros_like(observations.ss)

    return ss, pr

def boundElevations(observations):
    """ remove elevations outside of 0 or 85 degrees/1.48 rad
    """
    height=observations["sv_z"]-observations["z"]
    distance=np.linalg.norm(observations[["sv_x","sv_y"]].to_numpy()-observations[["x","y"]].to_numpy(),axis=1)
    elevation = np.arctan2(height,distance)
    return observations[(elevation>0) & (elevation <85/360*2*pi)]

def locateSatellites(points,map):
    """ generate an set of satellite observations without signal readings.
    Parameters
    ----------
    points : ReceiverPoints
        position of receiver
    
    Returns
    -------
    observations : Observations
        GNSS readings from receiver without a signal strength or pseudorange
    """
    SVIDS = np.array(np.char.add("E",np.char.zfill(np.arange(1,37).astype(str),2)).tolist()+np.char.add("G",np.char.zfill(np.arange(1,33).astype(str),2)).tolist()+np.char.add("R",np.char.zfill(np.arange(1,25).astype(str),2)).tolist()+np.char.add("C",np.char.zfill(np.arange(1,38).astype(str),2)).tolist())

    sv = SVIDS.repeat(points.shape[0])
    points_=pd.concat([points]* SVIDS.shape[0],ignore_index=True)
    data=GNSSData()

    wgs = data.satLocation(sv,points_.t)
    missing=np.isnan(wgs[:,0])
    points_=points_.loc[~missing]
    wgs = wgs[~missing,:]
    sv=sv[~missing]
    sv_points= map.clip(points_,wgs)
    
    return Observations(points_.x,points_.y,points_.z,points_.t,sv,*sv_points.T)