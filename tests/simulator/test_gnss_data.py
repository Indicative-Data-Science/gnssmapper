"""Unittests for the functions in svid_location, using the true data from 2020-02-11."""

import unittest
from simulator.gnss_data import *
import numpy.testing as npt
import numpy as np
import os
import pandas as pd
import copy

class TestHelperFunctions(unittest.TestCase):
    def test_utc_to_doy(self) -> None:
        time=np.array([np.datetime64('2020-02-11','D')])
        self.assertEqual(utc_to_doy(time)[0],["2020042"])
        npt.assert_almost_equal(utc_to_doy(time)[1],[0.0])

    def test_doyToGPS(self) -> None:
        date = "2020042"
        self.assertIsInstance(doyToGPS(date), dict)
        self.assertDictEqual(doyToGPS(date), {'week': 2092, 'day': 2})

    def test_estimateMeasurementTime(self):
        time=np.array([np.datetime64(int(38000000/.299792458),'ns'),np.datetime64(int(22000000/.299792458),'ns')])
        svid=np.array(["C01","G01"])
        npt.assert_almost_equal(np.array(estimateMeasurementTime(time,svid),dtype=float),np.array([0,0]))

    def test_poly_lagrange(self):
        time= np.arange(-0.3,0.7,0.1)
        x = [t/0.7  for t in time]
        y = [t *2/0.7 for t in time]
        z = [t**2/0.07 for t in time]
        scale=[0.7,1,2,1]
        data = {'gpstime': time,'x': x, 'y':y, 'z':z}
        poly = pd.DataFrame(data)
        test1 = poly_lagrange(3,poly)
        test2 = poly_lagrange(5,poly)
        self.assertAlmostEqual(test1[0],0)
        self.assertAlmostEqual(test2[0],0.2)
        self.assertAlmostEqual(test1[1]['lb'],-0.3)
        self.assertAlmostEqual(test2[1]['ub'],0.6)
        npt.assert_almost_equal(test1[1]['mid'],[0,0,0,0])
        npt.assert_almost_equal(test1[1]['scale'],scale)
        npt.assert_almost_equal(test1[1]['x'],[0,1,0,0,0,0,0,0])
        npt.assert_almost_equal(test1[1]['y'],[0,1,0,0,0,0,0,0])
        npt.assert_almost_equal(test1[1]['z'],[0,0,7,0,0,0,0,0])
   
class TestSVIDLocation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sp3 = getSP3File('2020042')
        cls.example_orbits = getOrbits(cls.sp3)
        cls.gnssdata=GNSSData()
        cls.truncated_orbits=cls.example_orbits.loc[np.isin(cls.example_orbits['svid'],["G18","G14"]) & (cls.example_orbits['utctime'] <= np.datetime64("2020-02-11T02:00:00"))].reset_index(drop=True)
        cls.poly=createLagrangian(cls.truncated_orbits)
        for sv,dic in cls.poly.items():
            cls.gnssdata.orbits['2020042'][sv]= OrderedDict(sorted(dic.items()))


    def test_getSP3File(self) -> None:
        esm_path = 'ESA0MGNFIN_2020' + '042' + '0000_01D_05M_ORB.SP3.gz'
        local = str(os.getcwd())+'/data/sp3/' + esm_path
        self.assertTrue(os.path.exists(local))
        self.assertIsInstance(self.sp3, str)

    def test_getOrbits(self) -> None:
        self.assertIsInstance(self.example_orbits, pd.DataFrame)
        npt.assert_equal(list(self.example_orbits.columns),
                         ['epoch', 'utctime', 'svid', 'x', 'y', 'z', 'clockerror'])
        self.assertEqual(self.example_orbits['utctime'].dtype, np.dtype('datetime64[ns]'))

    def test_createLagrangian(self):
        self.assertCountEqual(list(self.gnssdata.orbits['2020042'].keys()),["G18","G14"])
        self.assertEqual(len(self.gnssdata.orbits['2020042']['G18']),len([3,7,11,15,19,20]))
        
    def test_setup(self):
        empty=GNSSData()
        self.assertEqual(empty.orbits,{})

    
    def test_updateOrbits(self):
        a=copy.deepcopy(self.gnssdata.orbits)
        self.gnssdata.updateOrbits(np.array([]))
        self.assertDictEqual(a,self.gnssdata.orbits)
    
    def test_locateSatellite(self):
        time=np.array([np.datetime64('2020-02-11T01:00:00','ns')])
        time_check =np.arange(np.datetime64('2020-02-11T00:15:00','ns'),np.datetime64('2020-02-11T01:40:00','ns'),np.timedelta64(20,'m'))
        time_check= np.append(time_check,np.datetime64('2020-02-11T01:40:00','ns'))
        gpstime=utc_to_doy(time)
        entry=self.truncated_orbits.loc[(self.truncated_orbits['svid']=="G18") & (self.truncated_orbits['utctime']==time[0]),:].reset_index(drop=True)
        npt.assert_almost_equal(list(self.gnssdata.orbits["2020042"]["G18"]),list(utc_to_doy(time_check)[1]))
        predict=self.gnssdata.locateSatellite(gpstime[0][0],gpstime[1][0],"G18")
        npt.assert_almost_equal(predict,np.asarray([entry.x,entry.y,entry.z]).flatten(),decimal=3)

    def test_satLocation(self):
        time=np.array([np.datetime64('2020-02-11T01:00:00','ns')])
        predict=self.gnssdata.satLocation(np.array(["G18"]),time+np.timedelta64(int(22000000/.299792458),'ns'))
        entry=self.truncated_orbits.loc[(self.truncated_orbits['svid']=="G18") & (self.truncated_orbits['utctime']==time[0]),:].reset_index(drop=True)
        npt.assert_almost_equal(np.array(predict).flatten(),np.asarray([entry.x,entry.y,entry.z]).flatten(),decimal=0)

if __name__ == '__main__':
    unittest.main()





