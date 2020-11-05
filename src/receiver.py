import numpy as np
import pandas as pd
import math
import random
from shapely.geometry import Point, Polygon
from map import Map
""" 
=========================================
Receiver Simulator for GNSS Map data collection 
=========================================

This module simulates different collection processes for obtaining GNSS data


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


class ReceiverEmulator:
    def __init__(self, map, day, num_samples=1000,receiver_offset=1.):
        """
        Parameters
        ----------
        map : map object
        Contains a map of the area being sampled

        day : np.datetime64
        UTC date of sample

        num_samples : int
        default number of samples to be returned if not specified

        receiver_offset: float
        Offset (in metres) of receiver height from ground level

        """

        self.map = map
        self.day=day
        self.num_samples = num_samples
        self.receiver_offset=receiver_offset

    def point_process(self, polygon_: Polygon= None, num_samples=None, sampling_rate=None):
        """
        Parameters
        ----------
        num_samples : int
            number of samples to be returned

        polygon_ : Shapely Polygon object
           all of the samples will be inside the sampling polygon

        sampling_rate : int
            frequency of readings

        Returns
        -------
        points : ReceiverPoints
            points from outside buildings chosen uniformly at random.

        """
        if num_samples == None:
            num_samples = self.num_samples
        # recursive process to account for invalid points
        if num_samples ==0:
            return ReceiverPoints()

        minx, miny, maxx, maxy = self.map.bbox

        x = np.random.random(num_samples)* (maxx-minx) + minx
        y = np.random.random(num_samples)* (maxy-miny) + miny
        xy = np.column_stack((x,y))

        xy = xy[self.map.isOutside(xy)]
        if polygon_:
            xy = xy[self.contains_array(polygon_, xy)]

        z = self.map.groundLevel(xy) + self.receiver_offset
        if sampling_rate:
            t= np.arange(self.day, sampling_rate*len(z), sampling_rate, dtype='datetime64[s]')
        else:
            t = self.day + np.timedelta64(72000,'s') * np.random.random(len(z)) # from 00:00 to 20:00 to match the readings from 1 sp3 file and because there is a index error on the get_svid function

        p = ReceiverPoints(xy[:,0],xy[:,1],z,t)
        q = self.point_process(polygon_=polygon_, num_samples=num_samples - p.shape[0])

        return pd.concat([p,q], axis=0,ignore_index=True)

    def random_walk(self, polygon_, x0=None, y0=None, num_samples: int = 1000,
                    avg_speed: float = .1, sampling_rate: int = 5, prev_times: np.array = None,
                    time_bound: list = None):
        """
        Parameters
        ----------
        num_samples : int
           number of samples to be returned

        polygon_ : Shapely Polygon object
           all of the samples will be in the sampling polygon

        x0 : float
            starting x coordinate for the walk

        y0 : float
            starting y coordinate for the walk

        avg_speed : int
           mean parameter based on average movement for the random walk (m/s)

        sampling_rate : int
            frequency of readings

        time_bound : list
            list of the bounded min and max times for the walk e.g a one hour bound ay 9am:
            [np.timedelta64(9,'h'), np.timedelta64(1,'h')]

        prev_times : np.array
            used for recursion, appending previous times to the current walk

        Returns
        -------
        points : ReceiverPoints
            points from outside buildings chosen uniformly at random.

        """
        if num_samples == 0:
            return ReceiverPoints()

        # Initialisation
        if not x0:
            minx, miny, maxx, maxy = polygon_.bounds
            x0 = random.uniform(minx, minx + (maxx - minx))
            y0 = random.uniform(miny, miny + (maxy - miny))
            while not polygon_.contains(Point(x0, y0)):
                x0 = random.uniform(minx, minx + (maxx - minx))
                y0 = random.uniform(miny, miny + (maxy - miny))

        if time_bound:
            t = np.array([self.day + time_bound[0]], dtype='datetime64[s]')
        else:
            t = np.array([self.day], dtype='datetime64[s]')

        if prev_times:
            t = np.array(prev_times[-1], dtype='datetime64[s]')

        x, y = np.array([x0]), np.array([y0])
        tempx, tempy = x[-1], y[-1]
        secs = 0
        while len(x) != num_samples:
            orientation = np.random.randint(0, 360)
            step_size = avg_speed
            x_ = step_size * math.cos(math.radians(orientation))
            y_ = step_size * math.sin(math.radians(orientation))
            if polygon_.contains(Point(tempx + x_, tempy + y_)):
                tempx += x_
                tempy += y_
                if secs%sampling_rate == 0:
                    x = np.append(x, tempx)
                    y = np.append(y, tempy)
                    if time_bound:
                        if (t[-1] + np.timedelta64(sampling_rate,'s')) > (t[0] + time_bound[1]):
                            t[-1] = t[0]
                    t = np.append(t, t[-1] + np.timedelta64(sampling_rate, 's'))
                secs += 1

        xy = np.column_stack((x, y))
        xy = xy[self.map.isOutside(xy)]
        t = t[self.map.isOutside(xy)]
        z = self.map.groundLevel(xy) + self.receiver_offset
        p = ReceiverPoints(xy[:, 0], xy[:, 1], z, t)
        q = self.random_walk(polygon_=polygon_, x0=x[-1], y0=y[-1], num_samples=num_samples - p.shape[0],
                             prev_times=t, time_bound=time_bound)

        return pd.concat([p, q], axis=0, ignore_index=True)

    @staticmethod
    def contains_array(polygon: Polygon, points: np.array):
        enclosed = [polygon.contains(Point(point_)) for point_ in points]
        return np.array(enclosed)

# p1 = Polygon([(0, 0), (528020, 183020), (527990, 182990)])
# print(ReceiverEmulator(Map('../../map/box.txt'), np.datetime64('2020-02-02')).point_process(polygon_=p1))
