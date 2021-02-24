""" 

This module simulates different spatiotemporal processes for recording GNSS observations

"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box, Point, Polygon

# from scipy.stats import rice

import gnssmapper.common as cm
from gnssmapper.observations import observe
from gnssmapper.geo import fresnel, to_crs, is_outside,ground_level

_rng = np.random.default_rng()

def simulate(map_: gpd.GeoDataFrame, method:str, num_samples: int, start:pd.Timestamp,end:pd.Timestamp,method_args:dict=dict(),samplings_args:dict=dict()) -> gpd.GeoDataFrame:
    """Simulates observations made by a receiver.

    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        A map
    method : str
        process used to generate receiverpoints. Values are {'point_process','random_walk'}
    num_samples : int
        number of receiverpoints to simulate
    start : pd.Timestamp
        start boundary for observation time.
    end : pd.Timestamp
        end boundary for observation time.
    method_args : dict, optional
        method dependent arguments for the process of generating receiver points. 
    samplings_args : dict, optional
        arguments for the process of sampling observations. See gnssmapper.sim.sample for details.

    Returns
    -------
    gpd.GeoDataFrame
        A set of observations
    """
    if method not in {'point_process', 'random_walk'}:
        raise ValueError("method must be one of {'point_process','random_walk'}")
    if method == 'point_process':
        points = point_process(map_=map_,num_samples=num_samples,start=start,end=end,**method_args)    
    else:
        points = random_walk(map_=map_,num_samples=num_samples,start=start,end=end,**method_args)    
    
    
    observations = simulate_observations(map_, points,set(['C','E','G','R']),**samplings_args)
    return observations

def simulate_observations(map_: gpd.GeoDataFrame, points: gpd.GeoDataFrame,constellations: set[str] = set(['C','E','G','R']),**sampling_args) -> gpd.GeoDataFrame:
    """Generates a simulated set of observations from a receiverpoints dataframe

    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        Map used to simulate attenuation effects.
    points : gpd.GeoDataFrame
        gnss receiverpoints including:
            receiver position (as point geometry)
            time (utc format)
    constellations : set[str], optional
        constellations to be simulated. If not supplied it is assumed to be all 4 major constellations.

    Returns
    -------
    gpd.GeoDataFrame
        observations including:
        geometry (linestring from receiver in direction of satellite)
        time
        sv
        signal features
    """    
    observations = observe(points, constellations)
    observations = to_crs(observations,map_.crs)
    observations['fresnel']= fresnel(map_,observations.geometry)
    observations = sample(observations,**sampling_args)
    return observations

def sample(observations: gpd.GeoDataFrame,SSLB:float=10, mu_:float=35,msr_noise:float=5) -> gpd.GeoDataFrame:
    """Generates stochastic estimates of signal strength for an observations set.
    
    Utilises pre-calculated fresnel parameter.

    Parameters
    ----------
    observations : gpd.GeoDataFrame
    SSLB : float, optional
        lowest signal strength that returns a reading, by default 10
    mu_ : float, optional
        mean signal strength for LOS signal, by default 35
    msr_noise : float, optional
        Variance for the simulated signal strengths, by default 5

    Returns
    -------
    gpd.GeoDataFrame
        observations with 'Cn0DbHz' column of simulated signal strengths
   """
    
    """
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
    
    if 'fresnel' not in observations.columns:
        raise(AttributeError('fresnel column missing. Use gnssmapper.geo.fresnel to calculate.'))

    mus = mu_ - observations.fresnel
    ss_ = _rng.normal(mus,msr_noise)
    
    obs = observations.copy()
    obs['Cn0DbHz'] = np.where(ss_ > SSLB, ss_, np.nan)
    
    return obs.convert_dtypes()


def _xy_point_process(map_:gpd.GeoDataFrame,polygon:Polygon,num_samples:int) ->gpd.GeoSeries:
    """ Generates a geoseries of (2d) points outside map_ buildings and inside polygon"""
    minx, miny, maxx, maxy = polygon.bounds
    xy = np.empty(shape=(0,2),dtype=float)
    n = num_samples - xy.shape[0]

    while n > 0:
        p = _rng.random((n,2)) * np.array([[maxx-minx,maxy-miny]]) + np.array([[minx,miny]]) 
        points=gpd.GeoSeries(gpd.points_from_xy(p[:,0],p[:,1]),crs=map_.crs)
        outside = p[is_outside(map_,points,polygon),:]
        xy= np.vstack((xy,outside))
        n = num_samples - xy.shape[0]

    return gpd.GeoSeries(gpd.points_from_xy(xy[:,0],xy[:,1]),crs=map_.crs)


def point_process(map_:gpd.GeoDataFrame, num_samples:int, start:pd.Timestamp,end:pd.Timestamp, polygon: Polygon = Polygon(), receiver_offset:float=1.0) -> gpd.GeoDataFrame:
    """Generates a set of receiver locations using a random point process.

    Receiver locations are within the map boundaries and outside of buildings. 
    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        A gnssmapper map
    num_samples : int
        number of receiver locations to generate
    start : pd.Timestamp
        start boundary for observation time.
    end : pd.Timestamp
        end boundary for observation time.
    polygon : Polygon, optional
        A bounding polygon, by default empty.
    receiver_offset : float, optional
        The altitude of receiver location above ground level, by default 1.0

    Returns
    -------
    gpd.GeoDataFrame
        Receiverpoints
    """
    cm.check.map(map_)

    if num_samples <= 0:
        return gpd.GeoDataFrame()

    if polygon.is_empty:
        polygon= box(*map_.geometry.total_bounds)

    xy = _xy_point_process(map_, polygon, num_samples)
    z = ground_level(map_,xy) + receiver_offset
    t = start + (end-start) * pd.Series(_rng.random(num_samples))
    return gpd.GeoDataFrame({'time':t},geometry=gpd.points_from_xy(xy.x,xy.y,z),crs=xy.crs)

def random_walk(map_:gpd.GeoDataFrame, num_samples: int, start:pd.Timestamp,end:pd.Timestamp, polygon: Polygon = Polygon(), receiver_offset:float=1., avg_speed: float = .1, sampling_rate: int = 5) ->gpd.GeoDataFrame:
    """Generates a set of receiver locations using a random point process.

    Receiver locations are within the map boundaries and outside of buildings.

    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        A gnssmapper map
    num_samples : int
        number of receiver locations to generate
    start : pd.Timestamp
        start boundary for observation time.
    end : pd.Timestamp
        end boundary for observation time.
    polygon : Polygon, optional
        A bounding polygon, by default empty.
    receiver_offset : float, optional
        The altitude of receiver location above ground level, by default 1.0
    avg_speed : float, optional
        speed of random walk per second, by default .1
    sampling_rate : int, optional
        frequency of readings in seconds, by default 5

    Returns
    -------
    gpd.GeoDataFrame
        Receiverpoints
    """
    
    if num_samples <= 0:
        return gpd.GeoDataFrame()

    if polygon.is_empty:
        polygon= box(*map_.geometry.total_bounds)

    starting_point = _xy_point_process(map_,polygon,1)
    x, y, s = [starting_point.x[0]], [starting_point.y[0]], [0]
    tempx, tempy, temps = x[-1], y[-1], s[-1]

    while len(x) != num_samples:
        orientation = _rng.uniform(0,2 * np.pi)
        x_ = avg_speed * np.cos(orientation)
        y_ = avg_speed * np.sin(orientation)
        p = gpd.GeoSeries(Point(tempx + x_, tempy + y_),crs=map_.crs)
        if p.within(polygon)[0]:
            tempx += x_
            tempy += y_
            temps += 1
            if temps%sampling_rate == 0 and is_outside(map_,p)[0]:
                x.append(tempx)
                y.append(tempy)
                s.append(temps)

    xy = gpd.GeoSeries(gpd.points_from_xy(x,y),crs=map_.crs)
    z = ground_level(map_,xy) + receiver_offset

    time_range = (end - start).total_seconds()
    bounded_s = pd.to_timedelta(np.mod(s,time_range),unit='S')
    t = start + bounded_s
    return gpd.GeoDataFrame({'time':t},geometry=gpd.points_from_xy(xy.x,xy.y,z),crs=xy.crs)

