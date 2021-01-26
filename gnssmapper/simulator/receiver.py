import numpy as np
import pandas as pd
import math
import random
from shapely.geometry import box, Point
""" 
=========================================
Recording Process Simulator 
=========================================

This module simulates different spatiotemporal processes for recording GNSS observations

"""
# helper function to construct [num_samples,[x,y,z,t]] dataframe
def ReceiverPoints(x=None,y=None,z=None,t=None):
    #change length of empty arrays to non-zero if needed
    params=[x,y,z,t]
    n= max([0]+ [len(p) for p in params if p is not None])
    zero_params=[
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype='datetime64[s]')
    ]

    params=[p if p is not None else q for p,q in zip(params,zero_params)]

    return pd.DataFrame({
            'x':params[0],
            'y':params[1],
            'z':params[2],
            't':params[3]
            })


def xy_point_process(map,polygon,num_samples):
    """
        Parameters
    ----------
    map : map object
        Contains a map of the area being sampled
    
    polygon : Shapely Polygon object
        all of the samples will be inside the sampling polygon

    num_samples : int
        number of samples to be returned
    """
    minx, miny, maxx, maxy = polygon.bounds
    xy = np.empty(shape=(0,2),dtype=float)
    n = num_samples - xy.shape[0]

    while n > 0:
        p = np.random.random((n,2)) * np.array([[maxx-minx,maxy-miny]]) + np.array([[minx,miny]]) 
        outside = p[map.is_outside(p,polygon),:]
        xy= np.vstack((xy,outside))
        n = num_samples - xy.shape[0]

    return xy


def point_process(map, time_bound, num_samples=1000, polygon = None, receiver_offset=1.):
    """
    Parameters
    ----------
    map : map object
        Contains a map of the area being sampled
    
    time_bound : list
        list of the start and end times for the sample in  np.datetime64

    num_samples : int
        number of samples to be returned

    polygon : Shapely Polygon object
        all of the samples will be inside the sampling polygon. uses the map bounding box if no polygon given.

    receiver_offset: float
        Offset (in metres) of receiver height from ground level

    Returns
    -------
    points : ReceiverPoints
        points from outside buildings chosen uniformly at random.

    """
    
    if num_samples ==0:
        return ReceiverPoints()

    if polygon is None:
        polygon= box(*map.bbox)

    xy = xy_point_process(map,polygon,num_samples)
    z = map.ground_level(xy) + receiver_offset
    t = time_bound[0] + (time_bound[1]-time_bound[0]) * np.random.random(num_samples) # from 00:00 to 20:00 to match the readings from 1 sp3 file and because there is a index error on the get_svid function
    return ReceiverPoints(xy[:,0],xy[:,1],z,t)


def random_walk(map, time_bound, num_samples: int = 1000, polygon = None, receiver_offset=1., avg_speed: float = .1, sampling_rate: int = 5):
    """
    Parameters
    ----------
    map : map object
        Contains a map of the area being sampled
    
    time_bound : list
        list of the start and end times for the sample in  np.datetime64
    
    num_samples : int
        number of samples to be returned
        
    polygon : Shapely Polygon object
        all of the samples will be inside the sampling polygon

    receiver_offset: float
        Offset (in metres) of receiver height from ground level

    avg_speed : int
        mean parameter based on average movement for the random walk (m/s)

    sampling_rate : int
        frequency of readings in seconds


    Returns
    -------
    points : ReceiverPoints
        points from outside buildings based on a Random walk

    """
    if num_samples == 0:
        return ReceiverPoints()

    #initialisation
    if polygon is None:
        polygon= box(*map.bbox)

    starting_point = xy_point_process(map,polygon,1)
    x, y, s = [starting_point[0,0]], [starting_point[0,1]], [0]
    tempx, tempy, temps = x[-1], y[-1], s[-1]

    while len(x) != num_samples:
        orientation = random.uniform(0,2 * math.pi)
        x_ = avg_speed * math.cos(orientation)
        y_ = avg_speed * math.sin(orientation)
        if polygon.contains(Point(tempx + x_, tempy + y_)):
            tempx += x_
            tempy += y_
            temps += 1
            if temps%sampling_rate == 0 and map.is_outside(np.array([[tempx,tempy]])):
                x.append(tempx)
                y.append(tempy)
                s.append(temps)

    xy = np.column_stack((x, y))
    z = map.ground_level(xy) + receiver_offset

    time_range = np.array(np.timedelta64(time_bound[1] - time_bound[0],'s'),dtype=int)
    bounded_s = np.mod(s,time_range)
    t = time_bound[0] + np.array([np.timedelta64(int(s),'s') for s in bounded_s])

    return ReceiverPoints(xy[:,0],xy[:,1],z,t)


