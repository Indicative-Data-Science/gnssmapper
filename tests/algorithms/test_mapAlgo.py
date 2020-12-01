"""Unittests for the functions in svid_location, using the true data from 2020-02-11."""

import unittest
from algorithms.mapAlgo import *
from simulator.gnss import Observations
import simulator.map as mp
import numpy.testing as npt
import pandas.testing as pt
import numpy as np
import math

class TestMapAlgorithm(unittest.TestCase):
    def setUp(self):
        self.map_=mp.Map("data/map/canyon.txt")
        self.observations = Observations(x=np.array([527990.,528015.]), y = np.array([183005.,183005.]),z= np.array([5.,10.]),sv_x=np.array([528015.,528035.]), sv_y = np.array([183005.,183005.]),sv_z= np.array([5.,10.]),ss=np.array([1.,2.]) ) 
        self.algo = MapAlgorithm(self.map_,"tests/data/map_algo_data.gz")
    def test_generate_data(self) -> None:
        expected = pd.DataFrame(data=[[1.,5.,np.nan],[2.,np.nan,10.]],columns=['ss','0','1'],dtype=np.float64)
        pt.assert_frame_equal(generate_data(self.map_,self.observations),expected)

    def test_load_data(self) -> None:
        data = generate_data(self.map_,self.observations)
        save_data("tests/data/map_algo_data.gz",data)
        loaded_data = load_data("tests/data/map_algo_data.gz")
        pt.assert_frame_equal(data,loaded_data)

    def test_visible(self) -> None:
        expected=np.array([[1,5],[2,np.nan]])
        npt.assert_array_almost_equal(self.algo.visible('0'),expected)

    def test_reconstruct(self):
        pass

    def test_heights(self):
        self.algo.params['0']=[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]]
        expected=np.array([[2.0,5.0],[np.nan,np.nan]])
        npt.assert_array_almost_equal(self.algo.heights,expected)

    def test_starting_params(self):
        data=np.array([[1,5],[np.nan,10]])
        expected = [[0.8,0.1,1,0.2],[0.8,0.1,5,0.2]]
        npt.assert_almost_equal(starting_params(data),expected)

    def test_fit_online(self):
        data=np.array([[1,2,3,4],[1,1,10,10]])
        params=fit(data,[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]],True,batch=3,iterations=10)
        npt.assert_almost_equal(params.shape,(10,2,4))
        [npt.assert_almost_equal(params[i-1,:,:],params[i,:,:]) for i in range(1,10) if i not in {2,5,8,9}]
        params=fit(data,[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]],True,batch=5,iterations=10,SGD_batch=3)
        [npt.assert_almost_equal(params[i-1,:,:],params[i,:,:]) for i in range(1,10) if i not in {2,4,7,9}]
        
    def test_batch_index(self):
        data=np.array([0,1,2,3,np.nan]*10)
        idx=get_batch_indices(data,8)
        npt.assert_equal([9,19,29,39,49,50],idx)

    # def test_fill_array(self):
    #     missing=np.array([False,True,False,True,False])
    #     values=np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
    #     expected=np.array([[1,1,1,1],[1,1,1,1],[2,2,2,2],[2,2,2,2],[3,3,3,3]])
    #     start_missing=np.array([True,False,False,True,False])
    #     npt.assert_almost_equal(fill_array(start_missing,values),expected)
    #     npt.assert_almost_equal(fill_array(missing,values),expected)

    def test_fit_offline(self):
        map_=mp.Map("data/map/canyon.txt")
        alg_ = MapAlgorithm(map_,"data/algo_intersections/default.gz")
        data=alg_.visible('0')
        batch=80
        SGD_batch=50
        params=fit(data,[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]],True,batch=batch,iterations=1000)
        [npt.assert_almost_equal(params[i-1,:,:],params[i,:,:]) for i in range(1,300) if i not in {79,159,239,299}]
        params=fit(data,[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]],True,batch=225,iterations=300,SGD_batch=SGD_batch)
        [npt.assert_almost_equal(params[i-1,:,:],params[i,:,:]) for i in range(1,300) if i not in {49,99,149,199,224,274,299}]
        
    # def test_check_convergence(self):
    #     param_pass=[[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]],[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]]]
    #     param_fail=[[[0.8,1.,2.,0.2],[0.8,1.,2.,0.2]],[[0.8,1.,2.,0.2],[0.9,1.,2.,0.2]]]
    #     self.assertTrue(check_convergence(param_pass))
    #     self.assertFalse(check_convergence(param_fail))

class TestAlgorithmPerformance(unittest.TestCase):
    def test_performance(self):
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()





