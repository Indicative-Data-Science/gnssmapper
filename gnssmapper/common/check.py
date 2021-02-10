""" This module provides checks that gnssmapper objects are valid.

Objects {rays, receiver_points, observations, maps} are special geoDataFrames.
Not implemented as classes because of the difficulty of subclassing Pandas dataframes.
"""
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pygeos

from . import constants

def rays(rays: gpd.GeoSeries) -> None:
    # warnings
    crs(rays)
    # errors
    tests = {
        'Missing geometries': rays.is_empty.any(),
        'Expecting Linestrings': not rays.geom_type.eq("LineString").all(),
        'Missing z coordinates': not pygeos.has_z(
            pygeos.io.from_shapely(rays)
            ).all(),
        'More than 2 points in Linestring': np.not_equal(
            pygeos.count_coordinates(
                pygeos.io.from_shapely(rays)),
            2 * len(rays)).any(),
    }
    _raise(tests)
    return None


def receiverpoints(points: gpd.GeoDataFrame) -> None:
    # warnings
    crs(points.geometry)
    if 'svid' in points.columns:
        constellations(
            points['svid'], constants.supported_constellations)
    # errors
    tests = {
        'Missing geometries': points.geometry.is_empty.any(),
        'Expecting Points': not points.geom_type.eq("Point").all(),
        'Missing z coordinates': not pygeos.has_z(
            pygeos.io.from_shapely(points.geometry)
            ).all(),
        '"time" column missing or not datetime':
            (('time' not in points.columns) or
            (points['time'].dtype != "datetime64[ns]")),
    }
    _raise(tests)
    return None


def observations(obs: gpd.GeoDataFrame) -> None:
    if 'svid' in obs.columns:
        constellations(obs['svid'], constants.supported_constellations)
    rays(obs.geometry)
    # errors
    tests = {
        '"svid" column missing or not string':
            ('svid' not in obs.columns) or
            (obs['svid'].dtype != "object"),
        '"time" column missing or not datetime':
            ('time' not in obs.columns) or
            (obs['time'].dtype != "datetime64[ns]"),
    }
    _raise(tests)
    return None


def map(mp: gpd.GeoDataFrame) -> None:
    #warnings
    # crs(mp.geometry)
    tests = {
        '"height" column missing or not float':
            ('height' not in mp.columns) or
            (mp['height'].dtype != "float"),   
    }
    return None

def _raise(tests: dict) -> None:
    errors = [k for k, v in tests.items() if v]
    if errors != []:
        text = ', '.join(errors)
        raise AttributeError(text)
    return None


def crs(df: gpd.GeoSeries) -> None:
    """ Warns if crs is 2D and will be promoted for transforms"""
    return None


def constellations(svid: pd.Series, expected: set[str]) -> None:
    unsupported = set(svid.str[0].unique()) - expected
    if unsupported:
        warnings.warn(f'Includes unsupported constellations: {unsupported}')
    return None

def nanos(time: pd.Series) -> None:
    """ Warns if floats being used in for gps time"""
    if time.dtype == "float":
        warnings.warn("Potential rounding errors due to GPS time in nanoseconds input as float")
    return None