""" This module provides checks that 'objects' are valid.

Objects {rays, reciever_points, observations, maps} in gnssmapper are geoDataFrames which need certain properties. 
These have not been implemented as Classes because of the underlying difficulty of subclassing a Pandas dataframe.
"""
import geopandas as gpd
import numpy as np
import pygeos


def rays(rays: gpd.GeoSeries) -> bool:
    if not rays.geom_type.eq("Linestring").all():
        raise ValueError('Invalid rays (expecting Linestrings)')

    if rays.is_empty.any():
        raise ValueError('Missing rays')
    
    if np.not_equal(pygeos.count_coordinates(rays),2).any():
        raise ValueError('Invalid rays (more than 2 points)')

    if not pygeos



def check_valid_observations(obs: gpd.GeoDataFrame) -> bool:
    """Checks a geodataframe is a valid set of observations."""
    pass


def check_valid_receiverpoints(points: gpd.GeoDataFrame) -> None:
    """Checks a geodataframe is a valid set of receiver points."""
    # fatal errors
    if points.is_empty.any():
        raise ValueError('Missing receiver locations')
    if not points.geom_type.eq("Point").all():
        raise ValueError('Invalid receiver locations (expecting Points)')

    if points.z.is_na().any():
        raise ValueError('Missing z coordinate in receiver locations')

    if 'time' not in points.columns:
        raise ValueError('"time" column missing')
    if points.['time'].dtype != "datetime":
        raise ValueError('datatype of times column is not datetime')

    # warnings
    # if crs is 2d and will be promoted....
    if 'svid' in points.columns:
        check_constellations(
            points['svid'], constants.supported_constellations)

    return None

