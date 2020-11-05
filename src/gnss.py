"""Class to simulate GNSS data for a given location and date."""
import pandas as pd
import numpy as np
import datetime
from scipy.stats import rice, norm
from svid_location import getOrbits, getSP3File, getSVIDLocation

""" 
=========================================
GNSS Emulator for GNSS Map data collection 
=========================================

This module simulates GNSS observations of signal strength 

"""

# helper function to construct dataframe with correct structure
def Observations(x=None,y=None,z=None,t=None,svid=None,sv_x=None,sv_y=None,sv_z=None,ss=None,pr=None):
    #change length of empty arrays to non-zero if needed 
    params=[x,y,z,t,svid,sv_x,sv_y,sv_z,ss,pr]
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
            'ss':params[8],
            'pr':params[9]
            })

class GNSSEmulator:
    def __init__(self, map,day,SSLB=10):
        """ 
        Parameters
        ----------
        map : map object
        Contains a map of the area being sampled

        day : np.datetime64
        UTC date of sample

        SSLB : float
        lowest signal strength that returns a reading

        """
        self.map = map
        self.SSLB = 10
        self.day = day
        self.orbits = None
        self.svid_func = None
        self.setDay()
    
    def setDay(self):
        #this would be better moved to a svid location module so it becomes a class with methods
        sp3 = getSP3File(np.datetime_as_string(self.day,unit="D"))
        self.orbits = getOrbits(sp3)
        self.svid_func = getSVIDLocation(self.orbits)

    def observe(self,points, msr_noise):
        """Simulates a set of satellite observations
        Parameters
        ----------
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
        observations = self.observeSat(points)

        #model the signal
        observations.ss, observations.pr = self.modelSignal(observations, msr_noise)

        return observations

    def observeSat(self,points):
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

        #extend the points to match the number of svids
        sv_u = self.orbits['svid'].unique() 
        sv = sv_u.repeat(points.shape[0])
        points_=pd.concat([points]* sv_u.shape[0],ignore_index=True)
        
        wgs = self.getSatLocation(sv,points_.t)
        p=self.map.clip(points_,wgs)
        
        observations=Observations(points_.x,points_.y,points_.z,points_.t,sv,*p.T)
        #remove elevations outside of 0 or 85 degrees/1.48 rad
        height=observations["sv_z"]-observations["z"]
        distance=np.linalg.norm(observations[["sv_x","sv_y"]].to_numpy()-observations[["x","y"]].to_numpy(),axis=1)
        elevation = np.arctan2(height,distance)
        observations = observations[(elevation>0) & (elevation <1.48)]

        return observations

    def getSatLocation(self,sv,times):
        """ wraps the underlying svid function
        Parameters
        ----------
        sv : [n,] string array 
            satellite ids
        times: [n,] np.datetime64 array
            time of observation
        Returns
        -------
        location : [n,3] array 
            location in wgs84 cartesian  co-ords 
        """

        #the orbits array is being altered in place by the getSVIDLocation function to be number of seconds since GPS EPOCH. THis needs to be fixed but the below is a work around
        GPS_UTC_OFFSET = datetime.datetime(1980, 1, 6).replace(tzinfo=datetime.timezone.utc).timestamp()
        times_offset = np.array([time.timestamp() - GPS_UTC_OFFSET for time in times])
        # need to see if this can be vectorised
        wgs= np.array([self.svid_func(None,time*1e9, svid) for time,svid in zip(times_offset,sv)])

        return wgs

    def modelSignal(self,observations, msr_noise):
        """
        Parameters
        ----------
        observations : Observations
            satellite  and receiver locations

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
        mu_= 35
        fresnel = self.map.fresnel(observations)
        mus = mu_ - fresnel
        sigmas=np.ones_like(observations.ss)*msr_noise
        ss_ =norm.rvs(mus,sigmas)
        
        ss = np.where(ss_ > self.SSLB, ss_, np.nan)
        pr = np.zeros_like(observations.ss)

        return ss, pr



