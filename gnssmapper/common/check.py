""" This module provides checks that gnssmapper objects are valid.

Objects {rays, receiver_points, observations, maps} are special geoDataFrames.
Not implemented as classes because of the difficulty of subclassing Pandas dataframes.
"""
from typing import Union
import warnings


import geopandas as gpd
import numpy as np
import pandas as pd
import pygeos
import pyproj.crs

from gnssmapper.common.constants import supported_constellations

def rays(rays: gpd.GeoSeries) -> None:
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
    if 'svid' in points.columns:
        constellations(
            points['svid'], supported_constellations)
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
        constellations(obs['svid'], supported_constellations)
    rays(obs.geometry)
    # errors
    tests = {
        '"svid" column missing or not string':
            ('svid' not in obs.columns) or
            ((obs['svid'].dtype !="object") and(obs['svid'].dtype !='StringDtype')),
        '"time" column missing or not datetime':
            ('time' not in obs.columns) or
            (obs['time'].dtype != "datetime64[ns]"),
    }
    _raise(tests)
    return None


def map(map_: gpd.GeoDataFrame) -> None:

    tests = {
        'Missing geometries': map_.geometry.is_empty.any(),
        'Expecting Polygons': not map_.geom_type.eq("Polygon").all(),
        'Unexpected z coordinates':  pygeos.has_z(
            pygeos.io.from_shapely(map_.geometry)
            ).any(),
            '"height" column missing or not numeric':
            ('height' not in map_.columns) or
            (map_['height'].dtype != "float" and map_['height'].dtype != "int"),   
    }
    _raise(tests)
    return None

def _raise(tests: dict) -> None:
    errors = [k for k, v in tests.items() if v]
    if errors != []:
        text = ', '.join(errors)
        raise AttributeError(text)
    return None


def crs(crs_: pyproj.crs.CRS) -> None:
    """ Warns if crs is 2D and will be promoted for transforms"""
    if len(crs_.axis_info)==2:
        warnings.warn('2D crs used. Will be promoted for transforms.')
    return None


def constellations(svid: Union[pd.Series,set[str]], expected: set[str]) -> None:
    if isinstance(svid, pd.Series):
        present = set(svid.str[0].unique())
    else:
        present=set(svid)
    
    unsupported = present - expected
    if unsupported:
        warnings.warn(f'Includes unsupported constellations: {unsupported}')
    return None
