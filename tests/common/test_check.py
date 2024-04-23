"""Unittests for the functions in check, using example datasets."""

import unittest

import geopandas as gpd
import pandas as pd
import pandas.testing as pt
import shapely.wkt
from shapely import linestrings, points

import gnssmapper.common.check as check
import gnssmapper.common.constants as cn


class TestCheck(unittest.TestCase):
    def setUp(self):
        line = linestrings([[[0, 0, 0], [1, 2, 3]]])[0]
        point = points([[0, 0, 0]])[0]

        self.rays = gpd.GeoSeries([line] * 4, crs=cn.epsg_wgs84_cart)
        self.points = gpd.GeoSeries([point] * 4, crs=cn.epsg_wgs84_cart)
        self.d = {
            "time": pd.Series([pd.Timestamp(year=2020, month=1, day=1)] * 4),
            "svid": pd.Series(["G01", "E01", "R01", "C01"]),
        }

    def test_raise(self) -> None:
        test = {"this should be raised": 1, "but not this": 0}
        self.assertRaisesRegex(
            AttributeError, "this should be raised", check._raise, test
        )

    def test_crs(self) -> None:
        bng = self.points.set_crs("EPSG:27700", allow_override=True)
        self.assertWarnsRegex(UserWarning, "2D crs", check.crs, bng.crs)

    def test_constellations(self) -> None:
        valid_receiverpoints = gpd.GeoDataFrame(self.d, geometry=self.points)
        part = {"R", "C", "E"}
        self.assertWarnsRegex(
            UserWarning,
            "Includes unsupported constellations:",
            check.constellations,
            valid_receiverpoints.svid,
            part,
        )

    def test_rays(self) -> None:
        self.assertTrue(check.check_type(self.rays, "rays"))

    def test_obs(self) -> None:
        valid_obs = gpd.GeoDataFrame(self.d, geometry=self.rays)
        self.assertTrue(check.check_type(valid_obs, "observations"))

    def test_receiverpoints(self) -> None:
        valid_receiverpoints = gpd.GeoDataFrame(self.d, geometry=self.points)
        self.assertTrue(check.check_type(valid_receiverpoints, "receiverpoints"))

    def test_map(self) -> None:
        valid_map = gpd.GeoDataFrame(
            {"height": [10]},
            geometry=[
                shapely.wkt.loads(
                    "POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))"
                )
            ],
            crs="epsg:27700",
        )
        self.assertTrue(check.check_type(valid_map, "map", True))
