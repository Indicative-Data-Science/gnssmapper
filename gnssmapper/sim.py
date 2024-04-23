"""

This module simulates different spatiotemporal processes for recording GNSS observations

"""

from typing import Set

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import expon, levy, uniform
from shapely.geometry import Point, Polygon, box

import gnssmapper.common as cm
from gnssmapper.common.check import Map, Observations, ReceiverPoints
from gnssmapper.geo import fresnel, ground_level, is_outside, to_crs
from gnssmapper.observations import observe

# from scipy.stats import rice


_rng = np.random.default_rng()


def simulate(
    map_: Map,
    bounds: np.array,
    start: pd.Timestamp,
    end: pd.Timestamp,
    num_samples: int,
    cluster: str = "none",
    cluster_args: dict = dict(),
    receiver_offset: float = 1.0,
    sampling_args: dict = dict(),
) -> Observations:
    """Simulates observations made by a receiver.

    Parameters
    ----------
    map_ : Map
     bounds : np.array
        spatial bounds with minx,miny,maxx,maxy format
    start : pd.Timestamp
        lower time bound
    end : pd.Timestamp
        upper time bound
    num_samples : int
        number of receivers (parent process) to simulate
    cluster : str, optional
        type of child process, by default 'none'
    cluster_args : dict,optional
        passed to clustering (child) process, by default dict().
    receiver_offset : float, optional
        The altitude of receiver location above ground level, by default 1.0
    sampling_args : dict, optional
        [description], by default dict()

    Returns
    -------
    Observations
    """

    points = point_process(
        map_, bounds, start, end, num_samples, cluster, cluster_args, receiver_offset
    )
    observations = simulate_observations(
        map_, points, {"C", "E", "G", "R"}, **sampling_args
    )
    return observations


def simulate_observations(
    map_: Map,
    points: ReceiverPoints,
    constellations: set[str] = {"C", "E", "G", "R"},
    **sampling_args
) -> Observations:
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
    observations = to_crs(observations, map_.crs)
    observations["fresnel"] = fresnel(map_, observations.geometry)
    observations = sample(observations, **sampling_args)
    return observations


def sample(
    observations: Observations, SSLB: float = 10, mu_: float = 35, msr_noise: float = 5
) -> Observations:
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

    if "fresnel" not in observations.columns:
        raise (
            AttributeError(
                "fresnel column missing. Use gnssmapper.geo.fresnel to calculate."
            )
        )

    mus = mu_ - observations.fresnel
    ss_ = _rng.normal(mus, msr_noise)

    obs = observations.copy()
    obs["Cn0DbHz"] = np.where(ss_ > SSLB, ss_, np.nan)

    return obs.convert_dtypes()


def point_process(
    map_: Map,
    bounds: np.array,
    start: pd.Timestamp,
    end: pd.Timestamp,
    num_samples: int,
    cluster: str = "none",
    cluster_args: dict = dict(),
    receiver_offset: float = 1.0,
) -> ReceiverPoints:
    """Generates a set of receiver locations using a clustered point process.

    Each cluster represents a set of receiver measurements. Child process can vary (none,random or levy walk, or a guided walk)
    Receiver locations are only returned if outside of buildings.

    Parameters
    ----------
    map_ : Map
    bounds : np.array
        spatial bounds with minx,miny,maxx,maxy format
    start : pd.Timestamp
        lower time bound
    end : pd.Timestamp
        upper time bound
    num_samples : int
        number of receivers (parent process) to simulate
    cluster_args : dict
        passed to clustering (child) process, by default dict().
    cluster : str, optional
        type of child process, by default 'none'
    receiver_offset : float, optional
        The altitude of receiver location above ground level, by default 1.0

    Returns
    -------
    ReceiverPoints
    """
    cm.check.check_type(map_, "map", raise_errors=True)
    xy, t = _poisson_cluster(bounds, start, end, num_samples, cluster, cluster_args)
    points = gpd.GeoSeries(gpd.points_from_xy(xy[0, :], xy[1, :]), crs=map_.crs)
    outside = is_outside(map_, points, box(*bounds))
    z = ground_level(map_, points[outside]) + receiver_offset

    return gpd.GeoDataFrame(
        {"time": t[outside]},
        geometry=gpd.points_from_xy(xy[0, outside], xy[1, outside], z),
        crs=map_.crs,
    )


def _poisson_cluster(
    bounds: np.array,
    start: pd.Timestamp,
    end: pd.Timestamp,
    num_samples: int,
    cluster: str,
    cluster_args: dict,
) -> tuple:
    """Generates a set of locations using a poisson cluster process."""

    if cluster not in ["none", "random", "levy", "guided"]:
        raise ValueError("cluster must be one of none,random,levy", "guided")

    parent_xy = _poisson_point(bounds, num_samples)
    parent_time = start + (end - start) * uniform.rvs(size=(num_samples,))
    if cluster == "none":
        return np.array([parent_xy[0, :], parent_xy[1, :]]), parent_time
    elif cluster in ["random", "levy"]:
        length = cluster_args["duration"].astype("timedelta64[s]").astype(np.int64)
        durations = np.ceil(expon.rvs(size=(num_samples,), scale=length)).astype(
            np.int64
        )
        xy = np.concatenate(
            [
                _walk(parent_xy[:, i], d, cluster, cluster_args["speed"])
                for i, d in enumerate(durations)
            ],
            axis=1,
        )
        time = np.concatenate(
            [
                t + np.array(range(0, d)).astype("timedelta64[s]")
                for t, d in zip(parent_time, durations)
            ]
        )
    else:
        xy_list = [
            _guided_walk(
                parent_xy[:, i], cluster_args["endpoint"], cluster_args["speed"]
            )
            for i in range(num_samples)
        ]
        durations = [x.shape[1] for x in xy_list]
        xy = np.concatenate(xy_list, axis=1)
        time = np.concatenate(
            [
                t + np.array(range(0, d)).astype("timedelta64[s]")
                for t, d in zip(parent_time, durations)
            ]
        )
    return np.array([xy[0, :], xy[1, :]]), time


def _walk(starting_point: np.array, steps: int, type: str, speed: float) -> np.array:
    """Generates a series of 2d points following walk process"""
    if type not in ["levy", "random"]:
        raise ValueError("walk type must be levy or random")
    if type == "levy":
        lo = levy.rvs(size=(steps,), scale=speed)
    if type == "random":
        lo = np.ones((steps,)) * speed
    angle = uniform.rvs(size=(steps,), scale=2 * np.pi)
    x = starting_point[0] + np.cumsum(lo * np.cos(angle))
    y = starting_point[1] + np.cumsum(lo * np.sin(angle))
    return np.array([x, y])


def _poisson_point(bounds: np.array, num_samples: int) -> np.array:
    """Generates a series of 2d points following homogenous poisson process"""
    minx, miny, maxx, maxy = bounds
    x = minx + uniform.rvs(size=(num_samples,), scale=maxx - minx)
    y = miny + uniform.rvs(size=(num_samples,), scale=maxy - miny)

    return np.array([x, y])


def _guided_walk(
    starting_point: np.array, end_point: np.array, speed: float
) -> np.array:
    """Generates a series of 2d points going straght to end point"""
    dx = end_point[0] - starting_point[0]
    dy = end_point[1] - starting_point[1]
    distance = np.sum((np.array(end_point) - np.array(starting_point)) ** 2) ** 0.5
    steps = np.ceil(distance / speed).astype("int64")
    x = starting_point[0] + np.cumsum(np.ones((steps,)) * (speed * dx / distance))
    y = starting_point[1] + np.cumsum(np.ones((steps,)) * (speed * dy / distance))
    x[-1] = end_point[0]
    y[-1] = end_point[1]
    return np.array([x, y])
