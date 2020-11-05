import numpy as np
import shapely
from itertools import repeat,chain,compress
from shapely.wkt import loads

import pyproj
from osgeo import ogr, gdal, osr
import matplotlib.pyplot as plt


""" 
=========================================
RMap class for GNSS simulator 
=========================================

This module defines a Map class for use throughout the simulator.

"""

class Map:

    def __init__(self, map_location):
        self.map_location = map_location
        self.setMap()

    def setMap(self):
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
            #removes 3rd dim of poly
            e=poly.exterior
            ii=poly.interiors
            new_e=shapely.geometry.LineString([c[0:2] for c in e.coords])
            new_ii=[shapely.geometry.LineString([c[0:2] for c in i.coords]) for i in ii]
            return shapely.geometry.Polygon(new_e,new_ii)

        self.buildings = shapely.geometry.MultiPolygon([flatten(poly) for poly in poly3d]) # this converts to 2D
        self.heights = np.array([poly.exterior.coords[0][2] for poly in poly3d]) #this takes the height of first point in exterior line. no test for height validity (i.e. flat across building). could use a fn?
        buffer=10
        minx, miny, maxx, maxy = self.buildings.bounds
        self.bbox=(minx-buffer,miny-buffer,maxx+buffer,maxy+buffer)

    def plot(self):
            for building in self.buildings:
                plt.plot(*building.exterior.xy)
            plt.show()

    def isOutside(self,points):
        """  
        Parameters
        ----------
        points: [n,2] float array
            x-y point coordinates

        Returns
        -------
        isOutside: n,] boolean array 
            true if points are outside (including edge) of all building. 

        """
        mp=shapely.geometry.asMultiPoint(points)

        #could be sped up using a prepared poly object or an R-tree....
        k=[not self.buildings.intersects(p) for p in mp]
        
        return np.array(k)

    def groundLevel(self,points):
        """  
        Parameters
        ----------
        points: [n,2] array
            x-y point coordinates

        Returns
        -------
        ground level: [n,] float array 
            ground level for each given point. 

        """
        # so far doesn't do anything except assuming ground level is zero
        k=np.zeros((points.shape[0],))
                
        return k

    def clip(self,points,wgs):
        """ turns satellite positions into a BNG format
        Parameters
        ----------
        points : [n,..] ReceiverPoints
            position of receiver

        wgs : [n,3] array
            satellite positions in WGS84 cartesian coords

        Returns
        -------
        position: [n,3] array
            x,y coords on bounding box, elevation in radians 

        """
        EPSG_WGS84_CART = 4978
        EPSG_BNG = 27700
        # first the satellite wgs positions are bounded to a 100km box to ensure relative accuracy of crs transform
        wgs_ = self.bound(wgs)
        #next converted to BNG
        bng = self.reproject(wgs_, EPSG_WGS84_CART, EPSG_BNG)
        
        receiver = np.array(points[["x","y","z"]])
        rays = [shapely.geometry.LineString([r,s]) for r,s in zip(receiver.tolist(),bng.tolist())]
        box_=shapely.geometry.box(*self.bbox).exterior
        
        intersections=[box_ & ray for ray in rays]
        pos = np.array([intersection.coords for intersection in intersections]).squeeze()
        assert pos.shape==wgs.shape, "The clipping process has not generated 1 point for each satellite"
        
        # height=bng[:,2]-receiver[:,2]
        # distance=np.linalg.norm(bng[:,:2]-receiver[:,:2],axis=1)
        # elevation = np.arctan2(height,distance)
        # return np.column_stack((pos[:,:2],elevation))
        return pos

    @staticmethod
    def bound(wgs):
        """ bounds position to a 3D bounding box of 100km surrounding London
        Parameters
        ----------
        wgs : [n,3] array
            satellite positions in WGS84 cartesian coords

        Returns
        -------
        wgs : [n,3] array
            satellite positions on bounding box in WGS84 cartesian coords

        """
        ORIGIN = np.array([[3980000, -10000, 4970000]])  # London (WGS84 cartesian co-ordinates)
        BBOX_SIDE_LENGTH = 100000
        delta = wgs - ORIGIN
        scale = BBOX_SIDE_LENGTH / np.amax(np.abs(delta),axis=1)
        return ORIGIN+delta*np.expand_dims(scale,axis=1)

    @staticmethod
    def reproject(points, source, target):
        """reproject point coords given as EPSG refs
        Parameters
        ----------
        points : [n,3] array
            points in source coords

        source: string
            EPSG code
        
        target: code
            EPSG code

        Returns
        -------
        coordinates : [n,3] array
            points in target coords

        """
        transformer = pyproj.Transformer.from_crs(source,target, always_xy=True)
        pp = transformer.itransform(points)
        
        return np.array(tuple(pp))

    def isLos(self,observations):
        """  
        Parameters
        ----------
        observations : Observations
            GNSS readings from receiver without a signal strength or pseudorange

        Returns
        -------
        Boolean: 
            true if observation has a clear line of sight. 

        """
        los=np.ones((len(observations),), dtype=bool)
        receiver=observations[["x","y","z"]].to_numpy().tolist()
        sat=observations[["sv_x","sv_y","sv_z"]].to_numpy().tolist()
        rays = [shapely.geometry.LineString([r,s]) for r,s in zip(receiver,sat)]
        
        for building,height in zip(self.buildings,self.heights):
            idx= los==True
            n = sum(idx)
            los[idx]=intersects(list(compress(rays,idx)),[building]*n,np.ones(n,)*height)

        return los
    

    @staticmethod
    def get_height(observations,buildings):
        """
        Parameters
        ----------
        observations : [n] Observations
            GNSS readings from receiver without a signal strength or pseudorange

        buildings : [n] polygon 2d

        Returns
        -------
        height:  [n,] float array 
            height at which observation intersects building. Nan if no intersection or the svid is below ground level
        
        """
        receiver=observations[["x","y","z"]].to_numpy().tolist()
        sat=observations[["sv_x","sv_y","sv_z"]].to_numpy().tolist()
        # rays = (shapely.geometry.LineString([r,s]) if r[2]<s[2] else shapely.geometry.LineString() for r,s in zip(receiver,sat)) #ignores intersections where sat is below receiver height(i.e. below ground)
        rays = [shapely.geometry.LineString([r,s]) for r,s in zip(receiver,sat)]
        points=projected_intersection(rays,buildings)
        return np.array([np.nan if p.is_empty else p.z for p in points])

    def fresnel(self,observations):
        """  
        Parameters
        ----------
        observations : Observations
            GNSS readings from receiver without a signal strength or pseudorange

        Returns
        -------
        fresnel: [n,] array 
            the diffraction/fresnel zone attenuation. 

        """
        receiver=observations[["x","y","z"]].to_numpy().tolist()
        sat=observations[["sv_x","sv_y","sv_z"]].to_numpy().tolist()
        # rays = (shapely.geometry.LineString([r,s]) if r[2]<s[2] else shapely.geometry.LineString() for r,s in zip(receiver,sat)) #ignores intersections where sat is below receiver height(i.e. below ground)
        rays = [shapely.geometry.LineString([r,s]) for r,s in zip(receiver,sat)]
 
        return np.array([self.get_fresnel(ray) for ray in rays])

    def get_fresnel(self,ray):
        """ returns the fresnel attenuation for a ray using Epstein-Peterson method
        Parameters
        ----------
        ray : shapely LineString (3d)

        Returns
        -------
        float: fresnel diffraction attenuation

        """
        n=len(self.heights)

        points = intersection([ray]*n,self.buildings,self.heights)
        idx= np.nonzero([not p.is_empty for p in points])[0]
        if len(idx)==0:
            return 0

        sort_index= np.argsort(points[i].z for i in idx)
        idx=idx[sort_index]

        diffraction_points = [shapely.geometry.Point(points[i].x,points[i].y,self.heights[i]) for i  in  idx]
        start = chain([ray.boundary[0]],diffraction_points)   
        end = chain(diffraction_points,[ray.boundary[1]])
        next(end)
 
        ep_rays = [shapely.geometry.LineString([p,q]) for p,q in zip(start,end)]
        v = fresnel_parameter(ep_rays,diffraction_points)
        Jv = fresnel_integral(v) 
        return np.sum(Jv)

def fresnel_integral(v):
    return np.where( v<-0.7 , 0 , 6.9 + 20 * np.log(((v-0.1)**2 +1)**0.5 +v -0.1  ))
    
def fresnel_parameter(rays,diffraction_points):
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
    wavelength = 0.1903 # GPS L1 signal frequency of 1575.42 MHz
    distances = np.array([r.project(d) for r,d in zip(rays,diffraction_points)])
    nearest=(r.interpolate(d) for r,d in zip(rays,distances))
    
    heights = np.array([d.z-p.z for p,d in zip(nearest,diffraction_points)]) 
    
    v = np.where(distances==0, -np.inf,heights *( 2 / (wavelength * distances))**0.5)
    return v


def projected_intersection(rays,buildings):
    """ 3d intersection point between rays and buildings (assuming buildings have nonbounded heights). Empty Point if no intersection
    Parameters
    ----------
    rays : [n] shapely LineStrings (3d)

    buildings : [n] polygons (2d)

    Returns
    -------
    Points: [n] shapely Points (3d) as generator
        
    """
    if any(building.has_z for building in buildings): 
        raise ValueError("shapely cannot handle 3d intersections")
    intersections=(building.exterior & ray for building,ray in zip(buildings,rays))
    
    def lowest(points):
        coords=np.asarray(points)
        lowest=np.nonzero(coords[:,2]==min(coords[:,2]) )
        return points[lowest[0][0]]

    return (points if points.is_empty else lowest(points) for points in intersections)

def intersection(rays,buildings,heights):
    """ 3d intersection point (returns one with lowest height). Empty point if no intersect
    Parameters
    ----------
    rays : [n] shapely LineStrings (3d)

    buildings : [n] Polygons (2d)

    heights: [n,] float array
        building heights

    Returns
    -------
    Points: [n] shapely Points (3d)   
    """
    points=projected_intersection(rays,buildings)
    return [shapely.geometry.Point() if not p.is_empty and (p.z> h) else p for p,h in zip(points,heights)]
  
def intersects(rays,buildings,heights):
    """ a 3D intersection test
    Parameters
    ----------
    rays : [n] shapely LineStrings (3d)
        
    buildings : [n] polygons (2d)
        a polygon 

    height: float
        building height

    Returns
    -------
    Boolean:  [n,] float array 
    
    """
    points=projected_intersection(rays,buildings)
    return np.array([not p.is_empty and p.z<=h for p,h in zip(points,heights)])