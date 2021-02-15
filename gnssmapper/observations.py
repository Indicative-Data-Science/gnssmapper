""" 
Contains methods for generating observations from ReceiverPoints.

"""

import geopandas as gpd
import numpy as np
import pygeos

import gnssmapper.common as cm
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

    if ~constellations:
        if ~measured_constellations:
            raise ValueError(
                "Supported constellations cannot be inferred from receiverpoints and must be supplied")
        else:
            constellations = measured_constellations

    # Generate dataframe of all svids supported by receiver
    gps_time = cm.time.utc_to_gps(points['time'])
    sd = st.SatelliteData()
    svids = sd.name_satellites(gps_time).explode(
        ).dropna().reset_index(name='gps_time')
    svids = svids[svids['svid'].str[0].isin(constellations)]

    # locate the satellites
    sats = sd.locate_satellites(svids['svid'], svids['gps_time'])

    # revert to utc time
    sats['time'] = cm.time.gps_to_utc(sats['gps_time'])
    sats = sats.set_index(['time', 'svid'])

    # convert points into geocentric WGS and merge
    receiver = points.to_crs(
        cm.constants.epsg_satellites).set_index(['time', 'svid'])
    receiver = receiver.assign(
        x=receiver.geometry.x, y=receiver.geometry.y, z=receiver.geometry.z)
    observations = receiver.merge(sats)
    r = observations.loc[:, ["x", "y", "z"]].to_numpy().tolist()
    s = observations.loc[:, ["sv_x", "sv_y", "sv_z"]].to_numpy().tolist()
    rays = rays(r, s)

    obs.drop(columns=['x', 'y', 'z', 'sv_x', 'sv_y', 'sv_z', 'geometry'])
    obs = gpd.GeoDataFrame(obs, crs=cm.constants.epsg_satellites, geometry=rays)
    cm.check.observations(obs)
    
    # filter observations
    obs = filter_elevation(obs, cm.constants.minimum_elevation,
                           cm.constants.maximum_elevation)
    
    return obs


def rays(receivers: list, sats: list) -> gpd.LineString:
    """ Turns arrays of points into array of linestrings.

    The linestring is truncated towards the satellite. This is to avoid projected crs problems."""
    coords = [[tuple(r), tuple(s)] for r, s in zip(receivers, sats)]
    lines = pygeos.creation.linestrings(coords)
    short = pygeos.linear.line_interpolate_point(lines, cm.constants.ray_length)
    return short


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
    check_valid_rays(lines)
    ecef = lines.to_crs(cm.constants.epsg_wgs84_cart)
    lla = lines.to_crs(cm.constants.epsg_wgs84)

    # need to extract and check unit
    delta = pygeos.get_coordinates(ecef, include_z=True)
    lat = np.radians(lla.x)
    long_ = np.radians(lla.y)
    up = (np.cos(long_) * np.cos(lat), np.sin(long_) * np.cos(lat), np.sin(lat_))
    inner = np.sum(delta * up, axis=1)
    return np.degrees(np.arcsin(inner))
