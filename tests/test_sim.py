"""Unittests for the map module."""

import unittest

import geopandas as gpd
import numpy as np
import numpy.testing as npt
import pandas as pd
import shapely.geometry
import shapely.wkt

import gnssmapper.geo as geo
import gnssmapper.sim as sim


class TestPointProcess(unittest.TestCase):
    def setUp(self):
        self.map_box = gpd.GeoDataFrame(
            {"height": [10]},
            geometry=[
                shapely.wkt.loads(
                    "POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))"
                )
            ],
            crs="epsg:27700",
        )
        self.polygon = shapely.wkt.loads(
            "MULTIPOLYGON(((528020 183005, 528020 183000 ,528000 183000 , 528000 183005 ,528020 183005)))"
        )
        self.start = pd.Timestamp(np.datetime64("2020-09-01T09:00"))
        self.end = pd.Timestamp(np.datetime64("2020-09-01T10:00"))

    def test_walk(self):
        static = sim._walk([0, 0], 1000, "random", 0)
        npt.assert_array_almost_equal(static, np.zeros((2, 1000)))
        random = sim._walk([0, 0], 1, "random", 1)
        npt.assert_almost_equal(random[0] ** 2 + random[1] ** 2, 1)

    def test_poisson_point(self):
        points = sim._poisson_point([-10, 0, 0, 10], 1000)
        self.assertEqual((2, 1000), points.shape)
        self.assertGreaterEqual(np.min(points[0]), -10)
        self.assertGreaterEqual(np.min(points[1]), 0)
        self.assertGreaterEqual(0, np.min(points[0]))
        self.assertGreaterEqual(10, np.min(points[1]))

    def test_guided_walk(self):
        straight = sim._guided_walk([0, 0], [2, 1], 1.25**0.5)
        npt.assert_array_equal(straight, np.array([[1, 2], [0.5, 1]]))

    def test_poisson_cluster(self):
        points, time = sim._poisson_cluster(
            [-10, 0, 0, 10], self.start, self.end, 100, "none", dict()
        )
        self.assertEqual((2, 100), points.shape)
        self.assertEqual((100,), time.shape)
        self.assertGreaterEqual(np.min(points[0, :]), -10)
        self.assertGreaterEqual(np.min(points[1, :]), 0)
        self.assertGreaterEqual(0, np.min(points[0, :]))
        self.assertGreaterEqual(10, np.min(points[1, :]))
        self.assertTrue(np.all(time >= np.datetime64("2020-09-01T09:00")))
        self.assertTrue(np.all(time <= np.datetime64("2020-09-01T10:00")))

    def test_point_process(self):
        points = sim.point_process(
            self.map_box, self.polygon.bounds, self.start, self.end, 100
        )
        self.assertTrue(np.all(points["time"] >= np.datetime64("2020-09-01T09:00")))
        self.assertTrue(np.all(points["time"] <= np.datetime64("2020-09-01T10:00")))
        self.assertTrue(np.all(s.z > 0 for s in points.geometry))

        self.assertEqual(self.map_box.crs, points.crs)
        self.assertTrue(np.all(geo.is_outside(self.map_box, points.geometry)))
        self.assertTrue(all([self.polygon.contains(p) for p in points.geometry]))

    def test_sample(self) -> None:
        obs = pd.DataFrame({"fresnel": np.arange(0, 36, 35).repeat(10000)})
        SSLB = 10
        msr_noise = 1
        mu_ = 45
        obs = sim.sample(obs, SSLB, mu_, msr_noise)
        self.assertEqual(len(obs["Cn0DbHz"]), 20000)
        self.assertTrue(obs["Cn0DbHz"].min() >= SSLB)
        self.assertAlmostEqual(np.mean(obs["Cn0DbHz"][0:10000]), 45, places=0)
