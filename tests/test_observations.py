"""Unittests for the functions in observations"""

import unittest
import pandas.testing as pt
import numpy as np
import numpy.testing as npt
import pandas as pd
import geopandas as gpd
import geopandas.testing as gpt
import pygeos
import pyproj

from gnssmapper import observations
from gnssmapper import satellitedata
import gnssmapper.common as cm


class TestObservations(unittest.TestCase):
    def setUp(self) -> None:
        points = pd.DataFrame({'x': [527995], 'y': [183005], 'z':[0], 'time': [np.datetime64('2020-02-11T00:59:42', 'ns')]})
        self.points = gpd.GeoDataFrame(points, crs='EPSG:27700',geometry=gpd.points_from_xy(points['x'],points['y'],points['z']))
        cm.check.receiverpoints(self.points)

    def test_get_satellites(self) -> None:
        obs = observations._get_satellites(self.points, set(["C", "E", "G", "R"]))
        data = satellitedata.SatelliteData()
        data._update_orbits(["2020042"])
        self.assertTrue(np.all([l in list(data.orbits["2020042"].keys()) for l in list(obs.svid)]))
        wgs=data._locate_sat("2020042",3600*1e9,obs.svid[0])
        npt.assert_almost_equal(obs.loc[0,["sv_x","sv_y","sv_z"]],wgs,decimal=1)

    def test_get_multiple_satellites(self) -> None:
        points2 = pd.concat([self.points, self.points]).reset_index(drop=True)
        points2['svid']=['C06','C07']
        obs = observations._get_satellites(self.points, set(["C", "E", "G", "R"]))
        obs2 = observations._get_satellites(points2, set(["C", "E", "G", "R"]))
        pd.testing.assert_frame_equal(obs,obs2)
        

    def test_elevation(self) -> None:
        # 111319.458metres = 1 degree of longtitude  at 0 degrees latitude
        #expecting 45 degree elevation
        geometry = [pygeos.Geometry("LineString (0 0 0,0.01 0  1113.19458)"),
                    pygeos.Geometry("LineString (45 0 0,45.01 0 1113.19458)"),
                    pygeos.Geometry("LineString (90 0 0,90.01 0 1113.19458)")]
        lines = gpd.GeoSeries(geometry, crs=cm.constants.epsg_wgs84)
        self.assertTrue(np.all(44<observations.elevation(lines)) and np.all(46>observations.elevation(lines)))



