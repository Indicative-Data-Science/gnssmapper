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
        self.observations = Observations(x=np.array([527990,528015]), y = np.array([183005,183005]),z= np.array([5,10]),sv_x=np.array([528015,528035]), sv_y = np.array([183005,183005]),sv_z= np.array([5,10]),ss=np.array([1,2]) ) 
        
    def test_generate_data(self) -> None:
        expected = pd.DataFrame(data=[[1,5,np.nan],[2,np.nan,10]],columns=['ss'].extend(map_.buildingID))
        pt.assert_frame_equal(generate_data(self.map_,self.observations),b)

    def test_load_data(self) -> None:
        pass

    def test_visible(self) -> None:
        pass

    def test_reconstruct(self):
        pass

    def test_heights(self):
        pass

    def test_starting_params(self):
        pass

    def test_fit_online(self):
        pass

    def test_fill_array(self):
        pass

    def test_fit_offline(self):
        pass

    def test_check_convergence(self):
        pass


if __name__ == '__main__':
    unittest.main()





