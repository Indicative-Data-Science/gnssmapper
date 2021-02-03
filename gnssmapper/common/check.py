""" This module provides checks that 'objects' are valid.

Objects {rays, receiver_points, observations, maps} are special geoDataFrames.
Not implemented as Classes because difficulty of subclassing Pandas dataframe.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeos
import constants
import warnings


def rays(rays: gpd.GeoSeries) -> None:
    # warnings
    crs(rays)
    # errors
    tests = {
        'Missing geometries': rays.is_empty.any(),
        'Expecting Linestrings': not rays.geom_type.eq("Linestring").all(),
        'Missing z coordinates': not pygeos.has_z(rays).all(),
        'More than 2 points in Linestring': np.not_equal(
            pygeos.count_coordinates(rays), 2
        ).any(),
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
        'Expecting Points': points.geom_type.eq("Point").all(),
        'Missing z coordinates': points.z.is_na().any(),
        '"time" column missing or not datetime':
            (('time' not in points.columns) or
            (points['time'].dtype != "datetime")),
    }
    _raise(tests)
    return None


def observations(obs: gpd.GeoDataFrame) -> bool:
    if 'svid' in obs.columns:
        constellations(obs['svid'], constants.supported_constellations)
    rays(obs.geometry)
    # errors
    tests = {
        '"svid" column missing or not string':
            ('svid' not in obs.columns) or
            (obs['svid'].dtype != "string"),
        '"time" column missing or not datetime':
            ('time' not in obs.columns) or
            (obs['time'].dtype != "datetime"),
    }
    _raise(tests)
    pass


def _raise(tests: dict) -> None:
    errors = [k for k, v in tests.items() if v]
    if errors != []:
        text = ', '.join(errors)
        raise ValueError(text)
    return None


def crs(df: gpd.GeoSeries) -> None:
    """ Warns if crs is 2D and will be promoted for transforms"""
    return None


def constellations(svid: pd.Series, expected: set[str]) -> None:
    unsupported = set(svid.str[0].unique()) - expected
    if ~unsupported:
        warnings.warn(f'Includes unsupported constellations: {unsupported}')
    return None
