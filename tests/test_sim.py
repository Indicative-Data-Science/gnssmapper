"""Unittests for the map module."""
import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
import geopandas as gpd
import shapely.geometry
import shapely.wkt

import gnssmapper.geo as geo
import gnssmapper.sim as sim


class TestPointProcess(unittest.TestCase):
    def setUp(self):
        self.map_box = gpd.GeoDataFrame({'height': [10]},
            geometry=[shapely.wkt.loads("POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))")],
            crs="epsg:27700")
        self.polygon=shapely.wkt.loads('MULTIPOLYGON(((528020 183005, 528020 183000 ,528000 183000 , 528000 183005 ,528020 183005)))')
        self.start = pd.Timestamp(np.datetime64('2020-09-01T09:00'))
        self.end = pd.Timestamp(np.datetime64('2020-09-01T10:00'))

    def test_xy_process(self):
        points = sim._xy_point_process(self.map_box,self.polygon,1000)
        self.assertEqual(len(points), 1000)
        self.assertEqual(self.map_box.crs,points.crs)
        self.assertTrue(np.all(geo.is_outside(self.map_box,points)))
        self.assertTrue(all([self.polygon.contains(p) for p in points]))

    def test_point_process(self):
        points = sim.point_process(self.map_box,100,self.start,self.end,self.polygon)
        self.assertTrue(np.all(points["time"]>=np.datetime64('2020-09-01T09:00')))
        self.assertTrue(np.all(points["time"]<=np.datetime64('2020-09-01T10:00')))
        self.assertTrue(np.all(s.z>0 for s in points.geometry))

    def test_random_walk(self):
        points = sim.random_walk(self.map_box,100,self.start,self.end,self.polygon)
        self.assertTrue(np.all(geo.is_outside(self.map_box,points.geometry)))
        self.assertTrue(all([self.polygon.contains(p) for p in points.geometry]))
        self.assertTrue(np.all(points["time"]>=np.datetime64('2020-09-01T09:00')))
        self.assertTrue(np.all(points["time"]<=(np.datetime64('2020-09-01T10:00')+np.timedelta64(1,'ms'))))
        self.assertTrue(np.all(s.z>0 for s in points.geometry))

    def test_sample(self) -> None:
        obs=gpd.GeoDataFrame({'fresnel':np.arange(0,36,35).repeat(10000)})
        SSLB=10
        msr_noise=1
        mu_=45
        obs = sim.sample(obs,SSLB,mu_,msr_noise)
        self.assertEqual(len(obs['Cn0DbHz']),20000)
        self.assertTrue(obs['Cn0DbHz'].min()>=SSLB)
        self.assertAlmostEqual(np.mean(obs['Cn0DbHz'][0:10000]), 45, places=0)
        
        