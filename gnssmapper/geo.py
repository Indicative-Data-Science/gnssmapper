""" 

This module defines geometric methods that work in 3D and allow receiverpoints and observation objects to interact with a map

"""

import warnings

import geopandas as gpd
import numpy as np
import pygeos
import pandas as pd
import shapely.geometry
import math
import pyproj

from shapely.ops import transform
from typing import Union

import gnssmapper.common as cm

from shapely.wkt import loads
from itertools import chain, compress, cycle, repeat
import pyproj

def to_crs(df: Union[gpd.GeoDataFrame,gpd.GeoSeries], target: pyproj.crs.CRS) -> gpd.GeoDataFrame:
    """Reproject 3D geometry to target CRS.

    Bypasses geopandas to use shapely directly, avoiding bug of dropping Z coordinate when pygeos used.
    
    Parameters
    ----------
    geometry : gpd.GeoDataFrame
        series to be transformed
    target : pyproj.crs.CRS
        CRS to be transformed to

    Returns
    -------
    gpd.GeoDataFrame
        Transformed series.
    """
    def transform_geoseries(geometry):
        target_crs=pyproj.crs.CRS(target)
        cm.check.crs(target_crs)
        cm.check.crs(geometry.crs)
        transformer = pyproj.Transformer.from_crs(geometry.crs, target_crs)
        return gpd.GeoSeries([transform(transformer.transform,g) for g in geometry],crs=target)

    if isinstance(df,gpd.GeoDataFrame):
        transformed_geometry = transform_geoseries(df.geometry)
        return df.set_geometry(transformed_geometry,crs=target)
    else:
        return transform_geoseries(df)


def rays(receivers: list, sats: list) -> pygeos.Geometry:
    """ Turns arrays of points into array of linestrings.

    The linestring is truncated towards the satellite. This is to avoid projected crs problems."""
    coords = [[tuple(r), tuple(s)] for r, s in zip(receivers, sats)]
    lines = pygeos.creation.linestrings(coords)
    short = pygeos.linear.line_interpolate_point(lines, cm.constants.ray_length)
    short_coords = pygeos.coordinates.get_coordinates(short, include_z=True)
    coords=[[tuple(r),tuple(s)] for r,s in zip(receivers,short_coords)] 
    return pygeos.creation.linestrings(coords)


# def to_crs(map: gpd.GeoDataFrame,**kwargs):
#         """Transforms crs of geometry and height columns."""
#         print("hello")
#         return None

def is_outside(map_: gpd.GeoDataFrame, points: gpd.GeoSeries, polygon:shapely.geometry.Polygon=shapely.geometry.Polygon()) -> pd.Series:
    """ Returns boolean of whether points are outside of buildings.

    Element-wise testing that points do not intersect any of the geometries contained in a map. 

    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        A map
    points : gpd.GeoSeries
    
    polygon : shapely.geometry.Polygon, optional
        optional bounding polygon for points, by default shapely.geometry.Polygon()

    Returns
    -------
    pd.Series
        Boolean.
    """  
    
    
    # could be sped up using a prepared poly object or an R-tree....
    if polygon.is_empty:
        k = ~points.intersects(map_.geometry)
    else:
        k = points.within(polygon) & ~points.intersects(map_.geometry)

    return k

def ground_level(map_, points:gpd.GeoSeries) -> pd.Series:
    """ Returns ground level for each point. TO BE IMPLEMENTED """

    # so far doesn't do anything except assuming ground level is zero
    k = pd.Series(np.zeros((len(points),)))

    return k

def is_los(map_, rays:gpd.GeoSeries) -> pd.Series:
    """Returns boolean whether rays intersects buildings.

    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        A map
    rays : gpd.GeoSeries
        a collection of rays (linestrings consisting of 1 segment)
    
    Returns
    -------
    pd.Series
        true if ray has a clear line of sight.
    """   
    cm.check.rays(rays)
    
    los = np.ones((len(rays),), dtype=bool)
    

    for building, height in zip(map_.geometry, map_.height):
        idx = los == True
        n = sum(idx)
        los[idx] = intersects(list(compress(rays, idx)), [
                                building]*n, np.ones(n,)*height)

    return pd.Series(los)

def fresnel(map_, rays:gpd.GeoSeries) -> pd.Series:
    """The fresnel attenuation of a series of rays intersecting with a map.

    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        A map
    rays : gpd.GeoSeries
        a collection of rays (linestrings consisting of 1 segment)

    Returns
    -------
    pd.Series
        diffraction/fresnel zone attenuation
    """
    return pd.Series([get_fresnel(ray, map_.geometry, map_.heights) for ray in rays])

def projected_height(map_, rays:gpd.GeoSeries) -> pd.DataFrame:
    """The fresnel attenuation of a series of rays intersecting with a map.

    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        A map
    rays : gpd.GeoSeries
        a collection of rays (linestrings consisting of 1 segment)

    Returns
    -------
    pd.DataFrame
        height at which n observation intersects m projected building. Nan if no intersection
    """  
    
    b = chain(*repeat(map_.geometry, len(rays)))
    r = chain(*zip(*repeat(rays, len(map_.geometry))))

    heights = intersection_projected_height(r, b)
    heights = heights.reshape((-1, len(map_.geometry)))
    return pd.DataFrame(data=heights, columns=map_.index)


def drop_z(map_: gpd.GeoDataFrame):
    """Drops z attribute"""
    if pygeos.has_z(map_.geometry).any():
        warnings.warn("Geometry contains Z co-ordinates. Removed from Map3D (height attribute)")
    map_.geometry = pygeos.apply(map_.geometry, lambda x: x, include_z=False)
    return map_     

def is_inside(points, polygon):
    """  
    Parameters
    ----------
    points: [n] array of shapely points
        x-y point coordinates

    polygon: shapely polygon

    Returns
    -------
    is_outside: [n,] boolean array 
        true if points are inside a bounding polygon. 

    """
    mp = shapely.geometry.asMultiPoint(points)
    # could be sped up using a prepared poly object or an R-tree....

    k = [polygon.contains(p) for p in mp]

    return np.array(k)

def intersection_projected(rays, buildings):
    """ 3d intersection point between rays and buildings (always the lowest height and assuming buildings have nonbounded heights). Empty Point if no intersection
    Parameters
    ----------
    rays : (n) shapely LineStrings (3d) 

    buildings : (n) polygons (2d) 

    Returns
    -------
    Points: (n) shapely Points (3d) 

    """

    intersections = (building.exterior & ray for building,
                     ray in zip(buildings, rays))

    def lowest(points):
        coords = np.asarray(points)
        lowest = np.nonzero(coords[:, 2] == min(coords[:, 2]))
        return points[lowest[0][0]]

    return (points if points.is_empty else lowest(points) for points in intersections)


def intersection(rays, buildings, heights):
    """ 3d intersection point (returns one with lowest height). Empty point if no intersect
    Parameters
    ----------
    rays : (n) shapely LineStrings (3d)

    buildings : (n) Polygons (2d)

    heights: [n,] float array
        building heights

    Returns
    -------
    Points: [n] shapely Points (3d)   
    """
    points = intersection_projected(rays, buildings)
    return [shapely.geometry.Point() if not p.is_empty and (p.z > h) else p for p, h in zip(points, heights)]


def intersects(rays, buildings, heights):
    """ a 3D intersection test
    Parameters
    ----------
    rays : (n) shapely LineStrings (3d)

    buildings : (n) polygons (2d)

    heights: [n,] float array
        building heights

    Returns
    -------
    Boolean:  [n,] float array 

    """
    points = intersection_projected(rays, buildings)
    return np.array([not p.is_empty and p.z <= h for p, h in zip(points, heights)])


def intersection_projected_height(rays, buildings):
    """
    Parameters
    ----------
    rays : (n) shapely LineStrings (3d)

    buildings : (n) polygon 2d

    Returns
    -------
    height:  [n,] float array 
        height at which observation intersects building. Nan if no intersection

    """
    points = intersection_projected(rays, buildings)
    return np.array([np.nan if p.is_empty else p.z for p in points])


def get_fresnel(ray, buildings, heights):
    """ returns the fresnel attenuation across a set of buildings for a ray using Epstein-Peterson method
    Parameters
    ----------
    ray : shapely LineString (3d)

    buildings : [n] polygons (2d)

    heights: [n,] float array
        building heights

    Returns
    -------
    float: fresnel diffraction attenuation

    """
    points = intersection(repeat(ray), buildings, heights)
    idx = np.nonzero([not p.is_empty for p in points])[0]
    if len(idx) == 0:
        return 0

    sort_index = np.argsort(points[i].z for i in idx)
    idx = idx[sort_index]
    diffraction_points = [shapely.geometry.Point(
        points[i].x, points[i].y, heights[i]) for i in idx]
    start = chain([ray.boundary[0]], diffraction_points)
    end = chain(diffraction_points, [ray.boundary[1]])
    next(end)
    ep_rays = [shapely.geometry.LineString([p, q]) for p, q in zip(start, end)]
    v = fresnel_parameter(ep_rays, diffraction_points)
    Jv = fresnel_integral(v)
    return np.sum(Jv)


def fresnel_integral(v_array):
    def J(v):
        if v < -1:
            return 0
        if v < 0 and v >= -1:
            return 20 * math.log(0.5 - 0.62*v)
        if v < 1 and v >= 0:
            return 20 * math.log(0.5 * math.exp(-0.95*v))
        if v < 2.4 and v >= 1:
            return 20 * math.log(0.4 - (0.1184-(0.38-0.1*v)**2)**0.5)
        if v >= 2.4:
            return 20 * math.log(0.225/v)

    return np.array([-J(_) for _ in v_array])
    # return np.where( v<-0.7 , 0 , 6.9 + 20 * np.log(((v-0.1)**2 +1)**0.5 +v -0.1  )) #where did this come from?


def fresnel_parameter(rays, diffraction_points):
    """ returns the fresnel diffraction parameter (always as a positive)
    Parameters
    ----------
    rays : [n] list of shapely LineString (3d)

    diffraction_points: [n] list of Points (3d)
       diffraction point which the ray is rounding


    Returns
    -------
    fresnel diffraction parameters: [n,] float array

    """
    wavelength = 0.1903  # GPS L1 signal frequency of 1575.42 MHz
    distances = np.array([r.project(d)
                          for r, d in zip(rays, diffraction_points)])
    nearest_points = (r.interpolate(d) for r, d in zip(rays, distances))

    diffraction_distances = np.array(
        [d.z-p.z for p, d in zip(nearest_points, diffraction_points)])

    v = np.where(distances == 0, -np.inf, diffraction_distances *
                 (2 / (wavelength * distances))**0.5)
    return v

