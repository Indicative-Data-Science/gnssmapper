"""Unittests for the map module."""
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import numpy.testing as npt
import math

import gnssmapper.map as mp

class TestMapMethods(unittest.TestCase):
    def setUp(self):
        self.map_box = gpd.GeoDataFrame({'height': [10]},
            geometry=[shapely.wkt.loads("POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))")],
            crs="epsg:27700")

    def test_is_outside(self):
        point = self.map_box.buildings.representative_point()
        self.assertFalse(np.all(map_box.map.is_outside(point))

    def test_isground_level(self):
        point = shapely.geometry.box(*self.map_box.bbox).representative_point()
        xy=np.array([[point.x,point.y]])
        self.assertAlmostEqual(self.map_box.ground_level(xy),0)

    def test_projected_height(self):
        map_=mp.Map("tests/data/map/canyon.txt")
        observations = Observations(x=np.array([527990,528015]), y = np.array([183005,183005]),z= np.array([5,10]),sv_x=np.array([528015,528035]), sv_y = np.array([183005,183005]),sv_z= np.array([5,10]) )
        a = map_.projected_height(observations)
        b = pd.DataFrame(data=[[5,np.nan],[np.nan,10]],columns=map_.buildingID)
        pd.testing.assert_frame_equal(a,b)


class TestLos(unittest.TestCase):
    def setUp(self):
        self.map_box=mp.Map("tests/data/map/box.txt")
       
    def test_intersection(self):
        five = shapely.geometry.LineString([[527990,183005,0],[528020,183005,15]]) 
        point = mp.intersection([five],self.map_box.buildings,[10])
        self.assertAlmostEqual(np.array(point[0])[2],5)

    def test_intersection_projected_height(self):
        fifteen = shapely.geometry.LineString([[527990,183005,10],[528020,183005,25]])
        point = mp.intersection_projected_height([fifteen],self.map_box.buildings)
        self.assertAlmostEqual(point[0],15)

class TestFresnel(unittest.TestCase):
    def test_fresnel_integral(self):
        v=np.array([-1,0,1,2.4])
        o=np.array([-20*math.log(1.12),-20*math.log(0.5),-20*math.log(0.4-(0.1184-0.28**2)**0.5),-20*math.log(0.225/2.4)])
        npt.assert_almost_equal(mp.fresnel_integral(v),o)

    def test_fresnel_parameter(self):
        five = shapely.geometry.LineString([[527990,183005,5],[528020,183005,5]]) 
        point = shapely.geometry.Point([528000,183005,7])
        expected= 2 *( 2 / (0.1903 * 10))**0.5
        self.assertAlmostEqual(mp.fresnel_parameter([five],[point])[0],expected)

    def test_get_fresnel_single(self):
        map_box=mp.Map("tests/data/map/box.txt")
        five = shapely.geometry.LineString([[527990,183005,0],[528020,183005,15]]) 
        expected=mp.fresnel_integral([5 *( 2 / (0.1903 * 10))**0.5])
        self.assertAlmostEqual(mp.get_fresnel(five,map_box.buildings,map_box.heights),expected[0])

    def test_get_fresnel_multi(self):
        #not yet been tested
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
