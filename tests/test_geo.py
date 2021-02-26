"""Unittests for the map module."""
import unittest
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pygeos
import pyproj
import geopandas as gpd
import shapely.wkt
import numpy.testing as npt

import gnssmapper.common as cm
import gnssmapper.geo as geo


class TestObservationMethods(unittest.TestCase):
    def setUp(self):
        self.rays = gpd.GeoSeries([shapely.geometry.LineString([[527990, 183005, 0], [528020, 183005, 15]]),
                                       shapely.geometry.LineString([[527990, 183005, 10], [528020, 183005, 25]])],
                                        crs="epsg:27700")
    def test_rays(self) -> None:
        r = [[0, 0, 0], [1, 1, 1]]
        s = [[10000, 0, 0],[10001, 1, 1]]
        expected = [pygeos.Geometry("LineString (0 0 0,1000 0 0)"), pygeos.Geometry("LineString (1 1 1,1001 1 1)")]
        out=geo.rays(r,s)
        self.assertTrue(np.all(pygeos.predicates.equals(out,expected)))

    def test_to_crs(self) -> None:
        target = pyproj.crs.CRS(cm.constants.epsg_wgs84)
        transformed= geo.to_crs(self.rays,target)
        self.assertTrue(np.all(s.has_z for s in transformed))
        self.assertEqual(target,transformed.crs)

        df = gpd.GeoDataFrame(geometry = self.rays,crs=self.rays.crs)
        transformed_df = geo.to_crs(df,target)
        self.assertTrue(np.all(s.has_z for s in transformed_df.geometry))
        self.assertEqual(target,transformed_df.crs)


class TestShapelyMethods(unittest.TestCase):
    def setUp(self):
        self.building = shapely.wkt.loads("POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))")
       
    def test_intersection(self):
        five = shapely.geometry.LineString([[527990,183005,0],[528020,183005,15]]) 
        point = geo.intersection([five],[self.building],[10])
        self.assertAlmostEqual(np.array(point[0])[2],5)

    def test_intersection_projected(self):
        fifteen = shapely.geometry.LineString([[527990,183005,10],[528020,183005,25]])
        point = geo.intersection_projected([fifteen], [self.building])
        npt.assert_array_almost_equal(np.array(list(point)[0].coords).flatten(), [528000, 183005, 15])

        inside = shapely.geometry.LineString([[528005,183005,10],[528020,183005,25]])
        inside_point = geo.intersection_projected([inside], [self.building])
        npt.assert_array_almost_equal(np.array(list(inside_point)[0].coords).flatten(), [528010, 183005, 15])

        outside = shapely.geometry.LineString([[527990,183015,10],[528020,183015,25]])
        outside_point = geo.intersection_projected([outside], [self.building])
        self.assertTrue(list(outside_point)[0].is_empty)
        empty = shapely.geometry.LineString()
        empty_point = geo.intersection_projected([empty], [self.building])
        self.assertTrue(list(empty_point)[0].is_empty)


    def test_intersection_projected_height(self):
        fifteen = shapely.geometry.LineString([[527990,183005,10],[528020,183005,25]])
        point = geo.intersection_projected_height([fifteen],[self.building])
        self.assertAlmostEqual(point[0],15)

    def test_intersects(self):
        five = shapely.geometry.LineString([[527990, 183005, 0], [528020, 183005, 15]])
        fifteen = shapely.geometry.LineString([[527990, 183005, 10], [528020, 183005, 25]])
        rays = [five, fifteen]
        buildings = [self.building, self.building]
        heights=[10,10]
        npt.assert_array_almost_equal(geo.intersects(rays,buildings,heights),[True,False])
        

class TestFresnel(unittest.TestCase):
    def setUp(self):
        self.buildings = [shapely.wkt.loads("POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))")]

    def test_fresnel_integral(self):
        v=np.array([-1,0,1,2.4])
        o=np.array([-20*np.log(1.12),-20*np.log(0.5),-20*np.log(0.4-(0.1184-0.28**2)**0.5),-20*np.log(0.225/2.4)])
        npt.assert_almost_equal(geo.fresnel_integral(v),o)

    def test_fresnel_parameter(self):
        five = shapely.geometry.LineString([[527990,183005,5],[528020,183005,5]]) 
        point = shapely.geometry.Point([528000,183005,7])
        expected= 2 *( 2 / (0.1903 * 10))**0.5
        self.assertAlmostEqual(geo.fresnel_parameter([five],[point])[0],expected)

    def test_get_fresnel_single(self):
        five = shapely.geometry.LineString([[527990,183005,0],[528020,183005,15]]) 
        expected=geo.fresnel_integral([5 *( 2 / (0.1903 * 10))**0.5])
        self.assertAlmostEqual(geo.get_fresnel(five,self.buildings,[10]),expected[0])

    def test_get_fresnel_multi(self):
        #not yet been tested
        pass


class TestMapMethods(unittest.TestCase):
    def setUp(self):
        self.map_box = gpd.GeoDataFrame({'height': [10]},
            geometry=[shapely.wkt.loads("POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))")],
            crs="epsg:27700",index=[1])
        self.map_canyon =gpd.GeoDataFrame({'height': [10,10]},
            geometry=list(shapely.wkt.loads("MULTIPOLYGON(((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010)),((528030 183010, 528030 183000,528020 183000, 528020 183010,528030 183010)))")),
            crs="epsg:27700",index=[3,4])
        self.rays_box = gpd.GeoSeries([shapely.geometry.LineString([[527990, 183005, 0], [528020, 183005, 15]]),
                                       shapely.geometry.LineString([[527990, 183005, 10], [528020, 183005, 25]])],
                                        crs="epsg:27700",index=[1,2])
        self.rays_canyon = gpd.GeoSeries([shapely.geometry.LineString([(527990, 183005, 5), (528015, 183005, 5)]),
                                          shapely.geometry.LineString([(528015, 183005, 9), (528035, 183005, 9)])],
                                        crs="epsg:27700",index=[1,2])

    def test_map_to_crs(self):
        output = geo.map_to_crs(self.map_box, cm.constants.epsg_wgs84)
        cm.check.map(output)
        same = geo.map_to_crs(self.map_box, "epsg:27700")
        pdt.assert_frame_equal(self.map_box,same,check_dtype=False)
        
    def test_is_outside(self):
        point = self.map_box.geometry.representative_point()
        self.assertFalse(np.all(geo.is_outside(self.map_box, point)))
        point = self.map_canyon.geometry.representative_point()
        self.assertFalse(np.all(geo.is_outside(self.map_canyon, point)))
        point_series = gpd.GeoSeries(point.array,crs=self.map_canyon.crs,index=[10,11])
        pdt.assert_series_equal(geo.is_outside(self.map_canyon, point_series),pd.Series([False,False],index=[10,11]),check_names=False)
        

    def test_ground_level(self):
        point = self.map_box.geometry.representative_point()
        point_series = gpd.GeoSeries(point.array,crs=self.map_box.crs,index=[10])
        test = geo.ground_level(self.map_box, point)
        test_series = geo.ground_level(self.map_box, point_series)
        expected = pd.Series([0.0],index=[10])
        npt.assert_array_almost_equal(test, expected)
        pdt.assert_series_equal(test_series,expected,check_names=False)

    def test_is_los(self):
        pdt.assert_series_equal(geo.is_los(self.map_box, self.rays_box), pd.Series([True, False],index=[1,2]))
        pdt.assert_series_equal(geo.is_los(self.map_canyon, self.rays_canyon), pd.Series([False, False],index=[1,2]))

    def test_projected_height(self):  
        a = geo.projected_height(self.map_canyon,self.rays_canyon)
        b = pd.DataFrame(data=[[5,np.nan],[np.nan,9]],columns=[3,4],index=[1,2])
        pd.testing.assert_frame_equal(a, b)
        
    def test_fresnel(self):  
        a = geo.fresnel(self.map_box, self.rays_box)
        five = geo.fresnel_integral([5 *( 2 / (0.1903 * 10))**0.5])[0]
        b = pd.Series(data=[five,0],index=[1,2])
        pdt.assert_series_equal(a, b)