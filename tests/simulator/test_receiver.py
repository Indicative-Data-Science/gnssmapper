"""Unittests for the map module."""
import unittest
import numpy as np
import numpy.testing as npt
import shapely.geometry
from simulator import receiver as rec
from shapely.wkt import loads

class Test_receiver(unittest.TestCase):
    def setUp(self):
        self.map_box=mp.Map("map/box.txt")
        with open("simulator/test_polygon.txt") as f: 
            wkt_ = f.read()
        self.polygon=loads(wkt_)
        self.time_bounds

    def test_xy_process(self):
        points = rec.xy_point_process(self.map_box,self.polygon,1000)
        mp= shapely.geometry.Multipolygon(points)
        self.assertTrue(points.shape[0]==1000)
        self.assertTrue(all([self.map_box.isOutside(p) for p in mp]))
        self.assertTrue(all[self.polygon.contains(p) for p in mp])

    def test_point_process(self):
        points = rec.point_process(self.map_box,time_bould=1000,self.polygon)
        points = np.random.random((1000,3))* 200000 + np.array([[3880000, -110000, 4870000]])
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


class Test_map_methods(unittest.TestCase):
    def setUp(self):
        self.map_box=mp.Map("map/box.txt")

    def test_clip_can_be_reversed(self):
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

    def test_isOutside(self):
        point = self.map_box.buildings.representative_point()
        xy=np.array([[point.x,point.y]])
        self.assertFalse(self.map_box.isOutside(xy))

    def test_isgroundlevel(self):
        point = shapely.geometry.box(*self.map_box.bbox).representative_point()
        xy=np.array([[point.x,point.y]])
        self.assertAlmostEqual(self.map_box.groundLevel(xy),0)

class Test_isLos(unittest.TestCase):
    def setUp(self):
        self.map_box=mp.Map("map/box.txt")
       
    def test_intersection(self):
        five = shapely.geometry.LineString([[527990,183005,0],[528020,183005,15]]) 
        point = mp.intersection([five],self.map_box.buildings,[10])
        self.assertAlmostEqual(np.array(point[0])[2],5)

    def test_intersection_projected_height(self):
        fifteen = shapely.geometry.LineString([[527990,183005,10],[528020,183005,25]])
        point = mp.intersection_projected_height([fifteen],self.map_box.buildings)
        self.assertAlmostEqual(point[0],15)

# class Test_isFresnel(unittest.TestCase):


if __name__ == '__main__':
    unittest.main()
