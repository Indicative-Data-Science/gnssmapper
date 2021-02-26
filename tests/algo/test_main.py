"""Unittests for the functions in svid_location, using the true data from 2020-02-11."""

import unittest
import math

import numpy.testing as npt
import pandas.testing as pt
import geopandas as gpd
import numpy as np
import shapely.wkt

from gnssmapper.algo.main import *
from gnssmapper.algo.main import _heights
import gnssmapper.geo as geo


class TestMapAlgorithm(unittest.TestCase):
    def setUp(self):
        self.map_=gpd.GeoDataFrame({'height': [10,10]},
            geometry=list(shapely.wkt.loads("MULTIPOLYGON(((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010)),((528030 183010, 528030 183000,528020 183000, 528020 183010,528030 183010)))")),
            crs="epsg:27700",index=[3,4])
        rays = gpd.GeoSeries([shapely.geometry.LineString([(527990, 183005, 5), (528015, 183005, 5)]),
                                          shapely.geometry.LineString([(528015, 183005, 10), (528035, 183005, 10)])],
                                        crs="epsg:27700", index=[1, 2])
        self.observations = gpd.GeoDataFrame({'Cn0DbHz':[1.,2.]},geometry=rays,index=rays.index)
        

    def test_prepare_data(self) -> None:
        expected = pd.DataFrame(data=[[5.,np.nan,1.,],[np.nan,10.,2.]],columns=[3,4,'Cn0DbHz'],index=[1,2],dtype=np.float64)
        pt.assert_frame_equal(prepare_data(self.map_,self.observations),expected)



    def test_heights(self):
        params=[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]]
        expected=(2,3.5,5)
        npt.assert_array_almost_equal(_heights(params),expected)

    def test_fit_online(self):
        height = np.array([1, 2, 3, 4])
        ss = np.array([1, 1, 10, 10])
        params=fit(height,ss,iterations=4,batch_size=3,online=True, starting_params = [[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]],online_params={'batch_size':10})
        npt.assert_almost_equal(params.shape,(12,2,4))
        [npt.assert_almost_equal(params[i-1,:,:],params[i,:,:]) for i in range(1,10) if i not in {2,5,8,9}]
        params=fit(height,ss,iterations=2,batch_size=5,online=True, starting_params = [[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]],online_params={'batch_size':3})
        [npt.assert_almost_equal(params[i-1,:,:],params[i,:,:]) for i in range(1,10) if i not in {2,4,7,9}]
        

    def test_fit_offline(self):
        data= prepare_data(self.map_,self.observations)

        params=fit(data[3],data['Cn0DbHz'],iterations=4,batch_size=80,online=False, starting_params=[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]])
        [npt.assert_almost_equal(params[i-1,:,:],params[i,:,:]) for i in range(1,320) if i not in {79,159,239,319}]






