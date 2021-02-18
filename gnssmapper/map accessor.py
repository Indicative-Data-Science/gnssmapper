import warnings

import geopandas as gpd
import numpy as np
import pygeos
import pandas as pd
import shapely.geometry
import math

import gnssmapper.common as cm

from shapely.wkt import loads
from itertools import chain, compress, cycle, repeat
import pyproj

""" 
=========================================
Geometric methods and map class
=========================================
This module defines:
a set of geometric methods that work in 3D and can interface with receiver and observation objects (dataframes with a specified format)
a Map class for use throughout the simulator.

"""

@pd.api.extensions.register_dataframe_accessor("map")
class AccessorMap:
    def __init__(self, gpd_obj):
        # self._validate(gpd_obj)
        self._obj = gpd_obj
    
    # @staticmethod
    # def _validate(self):
    #     if obj.__class__.__name__ != "GeoDataFrame":
    #         raise ValueError("Must be a GeoPandas GeoDataFrame")
    #     if "height" not in obj.columns:
    #         raise AttributeError("Must have 'height' column")
    
    def to_crs(self,**kwargs):
        """Transforms crs of geometry and height columns."""
        print("hello")
        return None

    def is_outside(self, points: gpd.GeoSeries, polygon:shapely.geometry.Polygon=shapely.geometry.Polygon()) -> pd.Series:
        """ Returns boolean of whether points are outside of buildings.

        Parameters
        ----------
        points : gpd.GeoSeries
        polygon : shapely.geometry.Polygon, optional
            optional bounding polygon for points, by default shapely.geometry.Polygon()

        Returns
        -------
        pd.Series
            boolean
        """  
        
       
        # could be sped up using a prepared poly object or an R-tree....
        if polygon.is_empty:
            k = not points.intersects(self.geometry)
        else:
            k = points.within(polygon) and not points.intersects(self.geometry)

        return k

    def ground_level(self, points:gpd.GeoSeries) -> pd.Series:
        """ Returns ground level for each point. TO BE IMPLEMENTED """

        # so far doesn't do anything except assuming ground level is zero
        k = pd.Series(np.zeros((len(points),)))

        return k

    def is_los(self, rays:gpd.GeoSeries) -> pd.Series:
        """Returns boolean whether rays intersects buildings.

        Parameters
        ----------
        rays : gpd.GeoSeries
            a collection of rays (linestrings consisting of 1 segment)
        
        Returns
        -------
        pd.Series
            true if ray has a clear line of sight.
        """   
        cm.check.rays(rays)
        
        los = np.ones((len(rays),), dtype=bool)
        

        for building, height in zip(self.geometry, self.heights):
            idx = los == True
            n = sum(idx)
            los[idx] = intersects(list(compress(rays, idx)), [
                                  building]*n, np.ones(n,)*height)

        return pd.Series(los)

    def fresnel(self, rays:gpd.GeoSeries) -> pd.Series:
        """The fresnel attenuation of a series of rays intersecting with a map.

        Parameters
        ----------
        rays : gpd.GeoSeries
            a collection of rays (linestrings consisting of 1 segment)

        Returns
        -------
        pd.Series
            diffraction/fresnel zone attenuation
        """
        return pd.Series([get_fresnel(ray, self.geometry, self.heights) for ray in rays])

    def projected_height(self, rays:gpd.GeoSeries) -> pd.DataFrame:
        """The fresnel attenuation of a series of rays intersecting with a map.

        Parameters
        ----------
        rays : gpd.GeoSeries
            a collection of rays (linestrings consisting of 1 segment)

        Returns
        -------
        pd.DataFrame
            height at which n observation intersects m projected building. Nan if no intersection
        """  
     
        b = chain(*repeat(self.buildings, len(rays)))
        r = chain(*zip(*repeat(rays_var, len(self.buildings))))

        heights = intersection_projected_height(r, b)
        heights = heights.reshape((-1, len(self.buildings)))
        return pd.DataFrame(data=heights, columns=self.buildingID)




def drop_z(map_: gpd.GeoDataFrame):
    """Drops z attribute"""
    if pygeos.has_z(map_.geometry).any():
        warnings.warn("Geometry contains Z co-ordinates. Removed from Map3D (height attribute)")
    map_.geometry = pygeos.apply(map_.geometry, lambda x: x, include_z=False)
    return map_     


class Map:

    def __init__(self, map_location):
        self.map_location = map_location
        self.set_map()

    def set_map(self):
        """ Reads a WKT file containing 1 multipolygon with 3d coords. These are the buildings of our map.

        Attributes
        ----------
        buildings: Shapely Multipolygon object
            collection of 2D floorplans 

        heights: float array
            buiding heights  

        bbox: float tuple
            a bounding box of the region in (minx, miny, maxx, maxy) format



        """
        with open(self.map_location) as f:
            wkt_ = f.read()

        poly3d = loads(wkt_)

        def flatten(poly):
            # removes 3rd dim of poly
            e = poly.exterior
            ii = poly.interiors
            new_e = shapely.geometry.LineString([c[0:2] for c in e.coords])
            new_ii = [shapely.geometry.LineString(
                [c[0:2] for c in i.coords]) for i in ii]
            return shapely.geometry.Polygon(new_e, new_ii)

        self.buildings = shapely.geometry.MultiPolygon(
            [flatten(poly) for poly in poly3d])  # this converts to 2D
        if any(building.has_z for building in self.buildings):
            raise ValueError("shapely cannot handle 3d intersections")

        # this takes the height of first point in exterior line. no test for height validity (i.e. flat across building). could use a fn?
        self.heights = np.array([poly.exterior.coords[0][2]
                                 for poly in poly3d])
        buffer = 20
        minx, miny, maxx, maxy = self.buildings.bounds
        self.bbox = (minx-buffer, miny-buffer, maxx+buffer, maxy+buffer)
        self.buildingID = [str(i) for i in range(len(self.buildings))]

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

