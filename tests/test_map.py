"""Unittests for the map module."""
import unittest
import numpy as np
import pandas as pd
import shapely
import numpy.testing as npt
import math
import simulator.map as mp
from simulator.receiver import ReceiverPoints
from simulator.gnss import Observations

class TestBound(unittest.TestCase):

    def test_bound_all_moved_inside(self):
        points = np.random.random((1000,3))* 400000 + np.array([[3780000, -210000, 4770000]])
        bounded = mp.bound(points)
        def check_inside(point):
            ORIGIN = np.array([[3980000, -10000, 4970000]])
            BBOX_SIDE_LENGTH = np.array(100000)
            return np.all(point<= ORIGIN + BBOX_SIDE_LENGTH+1e-7) & np.all(point> ORIGIN - BBOX_SIDE_LENGTH - 1e-7)

        inside = np.apply_along_axis(check_inside,1,bounded)

        self.assertTrue( np.all(inside))

    def test_bound_does_not_move_boundary_points(self):
        points = np.random.random((1000,3))* np.array([[200000,200000,0]])+ np.array([[3880000, -110000, 4870000]])
        bounded = mp.bound(points)
        npt.assert_almost_equal(bounded,points)

    def test_bound_works_x(self):
        x = np.array([[4080000, -10000, 4970000]])
        y = np.array([[4180000, -10000, 4970000]])

        npt.assert_almost_equal(mp.bound(y),x)

    def test_bound_works_y(self):
        x = np.array([[3980000, -110000, 4970000]])
        y = np.array([[3980000, -210000, 4970000]])
        npt.assert_almost_equal(mp.bound(y),x)

    def test_bound_works_z(self):
        x = np.array([[3980000, -10000, 5070000]])
        y = np.array([[3980000, -10000, 5170000]])
        npt.assert_almost_equal(mp.bound(y),x)

    def test_bound_works_xyz(self):
        x = np.array([[4080000, -110000, 5070000]])
        y = np.array([[4180000, -210000, 5170000]])
        npt.assert_almost_equal(mp.bound(y),x)


class TestMapMethods(unittest.TestCase):
    def setUp(self):
        self.map_box=mp.Map("tests/data/map/box.txt")

    def test_clip_can_be_reversed(self):
        #this can't be reversed because the clip process is actually inaccurate for close points (immaterial difference for far away points)
        box_=shapely.geometry.box(*self.map_box.bbox).exterior
        xy_bng = [box_.interpolate(d,True) for d in np.random.random((1000,))]
        z_bng = [-100*200*z for z in  np.random.random((1000,))]
        points_bng = np.array([[p.x,p.y,z] for p,z in zip(xy_bng,z_bng)])
        internal_point=shapely.geometry.box(*self.map_box.bbox).representative_point()
        receiver=ReceiverPoints([internal_point.x]*1000,[internal_point.y]*1000, [1]*1000,)

        EPSG_WGS84_CART = 4978
        EPSG_BNG = 27700

        points_wgs=mp.reproject(points_bng,EPSG_BNG,EPSG_WGS84_CART)
        npt.assert_almost_equal(self.map_box.clip(receiver,points_wgs),points_bng)

    def test_is_outside(self):
        point = self.map_box.buildings.representative_point()
        xy=np.array([[point.x,point.y]])
        self.assertFalse(self.map_box.is_outside(xy))

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
