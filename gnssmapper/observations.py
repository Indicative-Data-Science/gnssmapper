""" 
Contains methods for generating observations from ReceiverPoints.

"""

import geopandas as gpd
import numpy as np
import pandas as pd

import gnssmapper.common as cm
from gnssmapper.geo import rays, to_crs
import gnssmapper.satellitedata as st


def observe(points: gpd.GeoDataFrame, constellations: set[str] = []) -> gpd.GeoDataFrame:
    """Generates a set of observations from a receiverpoints dataframe.

    Observations includes all above horizon svids, not only those measured in receiverpoints dataframe.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        gnss receiverpoints including:
            receiver position (as point geometry)
            time (utc format)

    constellations : set[str], optional
        constellations supported by gnss receiver. If not supplied it is inferred from the measured receiverpoints.

    Returns
    -------
    gpd.GeoDataFrame
        observations including:
        geometry (linestring from receiver in direction of satellite)
        time
        sv
        signal features
    """
    #preliminaries
    cm.check.receiverpoints(points)
    cm.check.constellations(constellations, cm.constants.supported_constellations)
    measured_constellations = set(points['svid'].str[0].unique())

    if not constellations:
        if not measured_constellations:
            raise ValueError(
                "Supported constellations cannot be inferred from receiverpoints and must be supplied")
        else:
            constellations = measured_constellations

    sats = _get_satellites(points,constellations)
    sats = sats.set_index(['time', 'svid'])

    # convert points into geocentric WGS and merge
    receiver = to_crs(points,cm.constants.epsg_satellites).set_index(['time', 'svid'])
    receiver = receiver.assign(
        x=receiver.geometry.x, y=receiver.geometry.y, z=receiver.geometry.z)
    obs = receiver.merge(sats)
    r = obs.loc[:, ["x", "y", "z"]].to_numpy().tolist()
    s = obs.loc[:, ["sv_x", "sv_y", "sv_z"]].to_numpy().tolist()
    lines = rays(r, s)

    obs = obs.drop(columns=['x', 'y', 'z', 'sv_x', 'sv_y', 'sv_z', 'geometry'])
    obs = gpd.GeoDataFrame(obs, crs=cm.constants.epsg_satellites, geometry=lines)
    cm.check.observations(obs)
    
    # filter observations
    obs = filter_elevation(obs, cm.constants.minimum_elevation,
                           cm.constants.maximum_elevation)
    
    return obs


def _get_satellites(points: gpd.GeoDataFrame, constellations: set[str]) -> pd.DataFrame:
    """ Dataframe of all svids visible to a set of points """
    cm.check.receiverpoints(points) 
    # Generate dataframe of all svids supported by receiver
    gps_time = cm.time.utc_to_gps(points['time'])
    sd = st.SatelliteData()
    svids = sd.name_satellites(gps_time).explode()
    svids = svids[svids.str[0].isin(constellations)]    
    svids = svids.dropna().rename_axis('gps_time').reset_index()

    # locate the satellites
    sats = sd.locate_satellites(svids['svid'], svids['gps_time'])

    # revert to utc time
    sats['time'] = cm.time.gps_to_utc(sats['gps_time'])
    return sats

def filter_elevation(observations: gpd.GeoDataFrame, lb: float, ub: float) -> gpd.GeoDataFrame:
    """Filters observations by elevation bounds.

    Parameters
    ----------
    observations : gpd.GeoDataFrame
        observations
    lb : float
        minimum elevation in degrees
    ub : float
        maximum elevation in degrees

    Returns
    -------
    gpd.GeoDataFrame:
        Filtered observations
    """
    if not 0 <= lb <= ub <= 90:
        raise ValueError(
            "Invalid elevation bounds. Must be between 0 and 90 degrees")
    e = elevation(observations.geometry)

    return observations.loc[lb <= e <= ub, :].copy()


def elevation(lines: gpd.GeoSeries) -> np.array:
    """ Returns elevation with respect to the wgs84 ellipsoid plane centred at the start of line. """
    cm.check.rays(lines)
    ecef = to_crs(lines,cm.constants.epsg_wgs84_cart)
    lla = to_crs(lines,cm.constants.epsg_wgs84)

    # extract unit vector in direction of satellite
    array = np.stack([np.array(a) for a in ecef],axis=0)
    delta = array[:, 1,:] - array[:, 0,:]
    delta = delta / np.linalg.norm(delta,axis=1,keepdims=True)
    
    #extract orthogonal unit vector at receiver location
    receiver_lla = np.stack([np.array(a)[0] for a in lla],axis=0) 
    lat = np.radians(receiver_lla[:,0])
    long_ = np.radians(receiver_lla[:,1])
    up = np.stack([ np.cos(long_) * np.cos(lat),
                    np.sin(long_)*np.cos(lat),
                    np.sin(lat)
                    ], axis=1)
                    
    # inner product                
    inner = np.sum(delta * up, axis=1)
    return np.degrees(np.arcsin(inner))


