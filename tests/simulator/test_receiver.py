"""Unittests for the map module."""
import unittest
import numpy as np
import numpy.testing as npt
import shapely.geometry
import simulator.map as mp
import simulator.receiver as rec
import shapely.wkt

class TestPoint(unittest.TestCase):
    def setUp(self):
        self.map_box=mp.Map("tests/data/map/box.txt")
        with open("tests/data/test_polygon.txt") as f: 
            wkt_ = f.read()
        self.polygon=shapely.wkt.loads(wkt_)
        self.time_bound=[np.datetime64('2020-09-01T09:00'),np.datetime64('2020-09-01T10:00')]

    def test_xy_process(self):
        points = rec.xy_point_process(self.map_box,self.polygon,1000)
        mp= shapely.geometry.asMultiPoint(points)
        self.assertTrue(points.shape[0]==1000)
        self.assertTrue(np.all(self.map_box.is_outside(mp)))
        self.assertTrue(all([self.polygon.contains(p) for p in mp]))

    def test_point_process(self):
        points = rec.point_process(self.map_box,self.time_bound,1000,self.polygon)
        self.assertTrue(np.all(points["t"]>=np.datetime64('2020-09-01T09:00')))
        self.assertTrue(np.all(points["t"]<=np.datetime64('2020-09-01T10:00')))
        self.assertTrue(np.all(points["z"]>0))

    def test_random_walk(self):
        points = rec.random_walk(self.map_box,self.time_bound,1000,self.polygon)
        mp= shapely.geometry.asMultiPoint(np.array(points[["x","y"]]))
        self.assertTrue(np.all(self.map_box.is_outside(mp)))
        self.assertTrue(all([self.polygon.contains(p) for p in mp]))
        self.assertTrue(np.all(points["t"]>=np.datetime64('2020-09-01T09:00')))
        self.assertTrue(np.all(points["t"]<=(np.datetime64('2020-09-01T10:00')+np.timedelta64(1,'ms'))))
        self.assertTrue(np.all(points["z"]>0))
        

if __name__ == '__main__':
    unittest.main()
