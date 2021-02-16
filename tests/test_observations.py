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
        data.update_orbits(["2020042"])
        self.assertTrue(np.all([l in list(data.orbits["2020042"].keys()) for l in list(obs.svid)]))
        wgs=data._locate_sat("2020042",3600*1e9,obs.svid[0])
        npt.assert_almost_equal(obs.loc[0,["sv_x","sv_y","sv_z"]],wgs,decimal=1)

    def test_rays(self) -> None:
        r = [[0, 0, 0], [1, 1, 1]]
        s = [[10000, 0, 0],[10001, 1, 1]]
        expected = [pygeos.Geometry("LineString (0 0 0,1000 0 0)"), pygeos.Geometry("LineString (1 1 1,1001 1 1)")]
        out=observations.rays(r,s)
        self.assertTrue(np.all(pygeos.predicates.equals(out,expected)))

    def test_to_crs_3d(self) -> None:
        target=pyproj.crs.CRS(cm.constants.epsg_wgs84)
        transformed = observations.to_crs_3d(self.points,target)
        self.assertTrue(pygeos.has_z(
            pygeos.io.from_shapely(transformed.geometry)
            ).all())
        
        transformed_series = observations.to_crs_3d(self.points.geometry,target)
        self.assertTrue(pygeos.has_z(
            pygeos.io.from_shapely(transformed_series.geometry)
            ).all())

    def test_elevation(self) -> None:
        # 111319.458metres = 1 degree of longtitude  at 0 degrees latitude
        #expecting 45 degree elevation
        geometry = [pygeos.Geometry("LineString (0 0 0,0 0.01 1113.19458)"),
                    pygeos.Geometry("LineString (0 45 0,0 45.01 1113.19458)"),
                    pygeos.Geometry("LineString (0 90 0,0 90.01 1113.19458)")]
        lines = gpd.GeoSeries(geometry, crs=cm.constants.epsg_wgs84)
        self.assertTrue(np.all(44<observations.elevation(lines)) and np.all(46>observations.elevation(lines)))
  
    # def test_rays(self) -> None:
    #     sats = observations._get_satellites(self.points, set(["C", "E", "G", "R"]))
    #     sats = sats.set_index(['time', 'svid'])

    #     # convert points into geocentric WGS and merge
    #     receiver = self.points.to_crs(
    #         cm.constants.epsg_satellites).set_index(['time', 'svid'])
    #     receiver = receiver.assign(
    #         x=receiver.geometry.x, y=receiver.geometry.y, z=receiver.geometry.z)
    #     observations = receiver.merge(sats)

    #     r = observations.loc[:, ["x", "y", "z"]].to_numpy().tolist()
    #     s = observations.loc[:, ["sv_x", "sv_y", "sv_z"]].to_numpy().tolist()
        

    # def test_model_signal(self) -> None:
    #     obs=Observations(fresnel=np.arange(0,36,35).repeat(10000))
    #     SSLB=15
    #     msr_noise=1
    #     mu_=45
    #     obs.ss,obs.pr=model_signal(obs,SSLB,mu_,msr_noise)
    #     self.assertEqual(obs.ss.shape[0],20000)
    #     self.assertEqual(obs.pr.shape[0],20000)
    #     self.assertTrue(min(obs.ss)>=SSLB)
    #     self.assertAlmostEqual(np.mean(obs.ss[0:10000]),45,places=0)

