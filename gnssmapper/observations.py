"""
Contains methods for generating observations from ReceiverPoints.

"""

import geopandas as gpd
import numpy as np
import pandas as pd

import gnssmapper.common as cm
import gnssmapper.satellitedata as st
from gnssmapper.common.check import Observations, ReceiverPoints
from gnssmapper.geo import coordinates, rays, to_crs, z


def observe(points: ReceiverPoints, constellations: set[str] = set()) -> Observations:
    """Generates a set of observations from a receiverpoints dataframe.

    Observations includes all above horizon svids, not only those measured in receiverpoints dataframe.

    Parameters
    ----------
    points : ReceiverPoints
        gnss receiverpoints including:
            receiver position (as point geometry)
            time (utc format)

    constellations : set[str], optional
        constellations supported by gnss receiver. If not supplied it is inferred from the measured receiverpoints.

    Returns
    -------
    Observations
        observations including:
        geometry (linestring from receiver in direction of satellite)
        time
        sv
        signal features


    """
    # preliminaries
    cm.check.check_type(points, "receiverpoints", raise_errors=True)
    cm.check.constellations(constellations, cm.constants.supported_constellations)

    measured_constellations = (
        set(points["svid"].str[0].unique()) if "svid" in points.columns else set()
    )

    if not constellations:
        if not measured_constellations:
            raise ValueError(
                "Supported constellations cannot be inferred from receiverpoints and must be supplied"
            )
        else:
            constellations = measured_constellations

    sats = _get_satellites(points, constellations)
    # convert GLONASS FCN codes
    points_osn = _convert_fcn(points)
    obs = _merge(points_osn, sats)

    # filter observations
    obs = filter_elevation(
        obs, cm.constants.minimum_elevation, cm.constants.maximum_elevation
    )
    obs.reset_index(drop=True, inplace=True)

    return obs


def _get_satellites(points: ReceiverPoints, constellations: set[str]) -> pd.DataFrame:
    """Dataframe of all svids visible to a set of points"""
    if len(points) == 0:
        return pd.DataFrame(columns=["svids", "time"])
    # Generate dataframe of all svids supported by receiver
    gps_time = cm.time.utc_to_gps(points["time"].drop_duplicates())
    sd = st.SatelliteData()
    svids = sd.name_satellites(gps_time).explode()
    valid_svids = set().union(
        *[cm.constants.supported_svids[c] for c in constellations]
    )
    svids = svids[svids.isin(valid_svids)]
    svids = svids.dropna().rename_axis("gps_time").reset_index()

    # locate the satellites
    sats = sd.locate_satellites(svids["svid"], svids["gps_time"])

    # revert to utc time
    sats["time"] = cm.time.gps_to_utc(sats["gps_time"])
    sats.drop(columns=["gps_time"], inplace=True)
    return sats


def _convert_fcn(points: ReceiverPoints) -> ReceiverPoints:
    # convert GLONASS satellites with a FCN code to Orbital Slot numbers (Elevation must be used to filter out duplicates)
    def is_fcn(x):
        return (x.str[0] == "R") & (x.str[1:].astype("int") >= 93)

    if "svid" in points.columns:
        points_ = points.copy()
        fcn = points_.loc[is_fcn(points["svid"]),]
        osn1 = fcn.copy()
        osn1["svid"] = fcn["svid"].map(cm.constants.fcn_to_osn).map(lambda x: x[1])
        points_.loc[is_fcn(points["svid"]), "svid"] = (
            fcn["svid"].map(cm.constants.fcn_to_osn).map(lambda x: x[0])
        )
        return pd.concat([points_, osn1], ignore_index=True)
    else:
        return points


def _merge(points: ReceiverPoints, sats: gpd.GeoDataFrame) -> Observations:
    # convert points into geocentric WGS and merge
    receiver = to_crs(points, cm.constants.epsg_satellites)
    receiver = receiver.assign(
        x=receiver.geometry.x, y=receiver.geometry.y, z=z(receiver.geometry)
    )
    location = receiver[["time", "x", "y", "z"]].drop_duplicates()
    obs = location.merge(sats, how="right", on=["time"])

    # add measurements if any taken
    if "svid" in receiver.columns:
        measurement = receiver.drop(columns=["geometry"])
        obs = obs.merge(measurement, how="left", on=["x", "y", "z", "svid", "time"])

    r = obs.loc[:, ["x", "y", "z"]].to_numpy().tolist()
    s = obs.loc[:, ["sv_x", "sv_y", "sv_z"]].to_numpy().tolist()
    lines = rays(r, s)

    obs = obs.drop(columns=["x", "y", "z", "sv_x", "sv_y", "sv_z"])
    obs = gpd.GeoDataFrame(obs, crs=cm.constants.epsg_satellites, geometry=lines)
    return obs


def filter_elevation(observations: Observations, lb: float, ub: float) -> Observations:
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
        raise ValueError("Invalid elevation bounds. Must be between 0 and 90 degrees")
    e = elevation(observations.geometry)
    valid = (lb <= e) & (e <= ub)

    return observations.loc[valid, :].copy()


def elevation(lines: rays) -> np.array:
    """Returns elevation with respect to the wgs84 ellipsoid plane centred at the start of line."""
    cm.check.check_type(lines, "rays", raise_errors=True)
    ecef = to_crs(lines, cm.constants.epsg_wgs84_cart)
    lla = to_crs(lines, cm.constants.epsg_wgs84)

    # extract unit vector in direction of satellite
    array = coordinates(ecef)
    delta = array[:, 1, :] - array[:, 0, :]
    delta = delta / np.linalg.norm(delta, axis=1, keepdims=True)

    # extract orthogonal unit vector at receiver location
    receiver_lla = coordinates(lla)[:, 0, :]
    # NB: coords are in xy order.
    lat = np.radians(receiver_lla[:, 1])
    long_ = np.radians(receiver_lla[:, 0])
    up = np.stack(
        [np.cos(long_) * np.cos(lat), np.sin(long_) * np.cos(lat), np.sin(lat)], axis=1
    )

    # inner product
    inner = np.sum(delta * up, axis=1)
    return np.degrees(np.arcsin(inner))
