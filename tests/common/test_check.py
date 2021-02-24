"""Unittests for the functions in check, using example datasets."""

import unittest
import pandas as pd
import pandas.testing as pt
import geopandas as gpd
import shapely.wkt
from pygeos import Geometry

import gnssmapper.common.check as check
import gnssmapper.common.constants as cn


class TestCheck(unittest.TestCase):
    def setUp(self):
        line = Geometry("LineString(0 0 0, 1 2 3)")
        point = Geometry("Point(0 0 0)")
        self.rays = gpd.GeoSeries([line] * 4, crs=cn.epsg_wgs84_cart)
        self.points = gpd.GeoSeries([point] * 4, crs=cn.epsg_wgs84_cart)
        self.d = {
            "time": pd.Series([pd.Timestamp(year=2020, month=1, day=1)] * 4),
            "svid": pd.Series(["G01", "E01", "R01", "C01"])
            }

    def test_raise(self) -> None:
        test = {"this should be raised": 1, "but not this": 0}
        self.assertRaisesRegex(AttributeError, "this should be raised", check._raise, test)
        
        
    def test_crs(self) -> None:
        bng = self.points.set_crs('EPSG:27700',allow_override=True)
        self.assertWarnsRegex(UserWarning,'2D crs',check.crs,bng.crs)
        

    def test_constellations(self) -> None:
        valid_receiverpoints = gpd.GeoDataFrame(self.d, geometry = self.points)
        part=set(["R","C","E"])
        self.assertWarnsRegex(UserWarning,'Includes unsupported constellations:',check.constellations,valid_receiverpoints.svid,part)

    def test_rays(self) -> None:
        self.assertIsNone(check.rays(self.rays))
    
    def test_obs(self) -> None:
        valid_obs = gpd.GeoDataFrame(self.d, geometry=self.rays)
        self.assertIsNone(check.observations(valid_obs))

    def test_receiverpoints(self) -> None:
        valid_receiverpoints = gpd.GeoDataFrame(self.d, geometry = self.points)
        self.assertIsNone(check.receiverpoints(valid_receiverpoints))

    def test_map(self) -> None:
        valid_map = gpd.GeoDataFrame({'height': [10]},
            geometry=[shapely.wkt.loads("POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))")],
            crs="epsg:27700")
        self.assertIsNone(check.map(valid_map))
