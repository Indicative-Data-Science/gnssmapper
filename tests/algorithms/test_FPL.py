"""Unittests for the functions in svid_location, using the true data from 2020-02-11."""

import unittest
from simulator.gnss import *
from simulator.receiver import ReceiverPoints
import simulator.map as mp
import numpy.testing as npt
import pandas.testing as pt
import numpy as np
import math

class TestGNSS(unittest.TestCase):
    def test_locate_satellites(self) -> None:
        points=ReceiverPoints(np.array([527995]),np.array([183005]),np.array([0]),np.array([np.datetime64('2020-02-11T01:00:00','ns')]))
        box=mp.Map("tests/data/map/box.txt")
        obs=locate_satellites(points,box)
        data=GNSSData()
        data.update_orbits(np.array(["2020042"]))
        self.assertTrue(np.all([l in list(data.orbits["2020042"].keys()) for l in list(obs.svid)]))
        wgs=np.array([data.locate_sat("2020042",3600*1e9,obs.svid[0])])
        npt.assert_almost_equal(obs.loc[0,["sv_x","sv_y","sv_z"]],box.clip(points,wgs),decimal=1)

    def test_model_signal(self) -> None:
        obs=Observations(fresnel=np.arange(0,36,35).repeat(10000))
        SSLB=15
        msr_noise=1
        mu_=45
        obs.ss,obs.pr=model_signal(obs,SSLB,mu_,msr_noise)
        self.assertEqual(obs.ss.shape[0],20000)
        self.assertEqual(obs.pr.shape[0],20000)
        self.assertTrue(min(obs.ss)>=SSLB)
        self.assertAlmostEqual(np.mean(obs.ss[0:10000]),45,places=0)

    def test_bound_elevations(self) -> None:
        obs=Observations([527990]*4,[183005]*4,[0]*4, [np.datetime64('2020-02-11T01:00:00','ns')]*4,["G01"]*4,[528020]*4,[183005]*4,[0,0.001,30*math.tan(84.9/360*2*math.pi),30*math.tan(85/360*2*math.pi)])
        pt.assert_frame_equal(bound_elevations(obs),obs.loc[1:2])



if __name__ == '__main__':
    unittest.main()





