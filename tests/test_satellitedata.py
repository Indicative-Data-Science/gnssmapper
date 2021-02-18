"""Unittests for the functions in svid_location, using the true data from 2020-02-11."""

import copy
from collections import OrderedDict
import importlib.resources
import os
import unittest


import numpy as np
import numpy.testing as npt
import pandas as pd

import gnssmapper.common.time as tm
import gnssmapper.satellitedata as sd
from gnssmapper import data


class TestHelperFunctions(unittest.TestCase):
    def test_poly_lagrange(self):
        time= np.arange(-0.3,0.7,0.1)
        x = [t/0.7  for t in time]
        y = [t *2/0.7 for t in time]
        z = [t**2/0.07 for t in time]
        scale=[0.7,1,2,1]
        data = {'tm': time,'x': x, 'y':y, 'z':z}
        poly = pd.DataFrame(data)
        test1 = sd._poly_lagrange(3,poly)
        test2 = sd._poly_lagrange(5,poly)
        self.assertAlmostEqual(test1[0],0)
        self.assertAlmostEqual(test2[0],0.2)
        self.assertAlmostEqual(test1[1]['lb'],-0.3)
        self.assertAlmostEqual(test2[1]['ub'],0.6)
        npt.assert_almost_equal(test1[1]['mid'],[0,0,0,0])
        npt.assert_almost_equal(test1[1]['scale'],scale)
        npt.assert_almost_equal(test1[1]['x'],[0,1,0,0,0,0,0,0])
        npt.assert_almost_equal(test1[1]['y'],[0,1,0,0,0,0,0,0])
        npt.assert_almost_equal(test1[1]['z'],[0,0,7,0,0,0,0,0])
   
class TestSP3Functions(unittest.TestCase):
    def test_SP3_filename(self) -> None:
        esm_path = 'ESA0MGNFIN_2020' + '042' + '0000_01D_05M_ORB.SP3.gz'
        local = data.sp3.__path__[0] +'/'+ esm_path
        self.assertTrue(os.path.exists(local))

    def test_get_SP3_file(self) -> None:
        sp3 = sd._get_sp3_file('2020042')
        self.assertIsInstance(sp3, str)

    def test_get_SP3_dataframe(self) -> None:
        sp3 = sd._get_sp3_file('2020042')
        example_orbits = sd._get_sp3_dataframe(sp3)
        self.assertIsInstance(example_orbits, pd.DataFrame)
        npt.assert_equal(list(example_orbits.columns),
                         ['epoch', 'date','time', 'svid', 'x', 'y', 'z', 'clockerror'])
        
    def test_setup(self):
        empty=sd.SatelliteData()
        self.assertEqual(empty.orbits,{})


class TestSVIDLocation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sp3 = sd._get_sp3_file('2020042')
        cls.example_orbits = sd._get_sp3_dataframe(cls.sp3)
        cls.SatelliteData=sd.SatelliteData()
        cls.truncated_orbits=cls.example_orbits.loc[np.isin(cls.example_orbits['svid'],["G18","G14"]) & (cls.example_orbits['time'] < 2*60*60*10**9)].reset_index(drop=True)
        cls.poly=sd._create_orbit(cls.truncated_orbits)
        for sv,dic in cls.poly.items():
            cls.SatelliteData.orbits['2020042'][sv]= OrderedDict(sorted(dic.items()))

 
    def test_create_orbit(self):
        self.assertCountEqual(list(self.SatelliteData.orbits['2020042'].keys()),["G18","G14"])
        self.assertEqual(len(self.SatelliteData.orbits['2020042']['G18']),len([3,7,11,15,19,20]))
        

    def test_update_orbits(self):
        a=copy.deepcopy(self.SatelliteData.orbits)
        self.SatelliteData._update_orbits(np.array([]))
        self.assertDictEqual(a,self.SatelliteData.orbits)
    
    def test_locate_sat_check_entries(self):
        entries = pd.Series(np.arange(np.datetime64('2020-02-11T00:14:42','ns'),np.datetime64('2020-02-11T01:39:42','ns'),np.timedelta64(20,'m')))
        entries = pd.Series(np.append(entries, np.datetime64('2020-02-11T01:39:42', 'ns')))
        entries = tm.gps_to_doy(tm.utc_to_gps(entries))
        npt.assert_almost_equal(list(self.SatelliteData.orbits["2020042"]["G18"]),list(entries['time']))
        
    def test_locate_sat(self):
        time = pd.Series([np.datetime64('2020-02-11T00:59:42', 'ns')])
        gpstime = tm.gps_to_doy(tm.utc_to_gps(time))
        entry=self.truncated_orbits.loc[(self.truncated_orbits['svid']=="G18") & (self.truncated_orbits['time']==gpstime['time'][0]),:].reset_index(drop=True)
        predict=self.SatelliteData._locate_sat(gpstime['date'][0],gpstime['time'][0],"G18")
        npt.assert_almost_equal(predict,np.asarray([entry.x,entry.y,entry.z]).flatten(),decimal=3)

    def test_locate_satellites(self):
        time = pd.Series([np.datetime64('2020-02-11T00:59:42', 'ns')])
        time = tm.utc_to_gps(time)
        time.name = 'gpstime'
        doy = tm.gps_to_doy(time)
        svid = pd.Series(["G18"],name='svid').convert_dtypes()
        predict=self.SatelliteData.locate_satellites(svid,time)
        entry=self.truncated_orbits.loc[(self.truncated_orbits['svid']=="G18") & (self.truncated_orbits['time']==doy['time'][0]),:].reset_index(drop=True)
        npt.assert_almost_equal(np.array(predict.loc[:,'sv_x':'sv_z']).flatten(),np.asarray([entry.x,entry.y,entry.z]).flatten(),decimal=0)

class TestNameSatellites(unittest.TestCase):
    def test_naming(self) -> None:
        time = tm.utc_to_gps(pd.Series([np.datetime64('2020-02-11T00:59:42', 'ns')]))
        data = sd.SatelliteData()
        out = data.name_satellites(time)
        self.assertSetEqual(
            set(out.iat[0]),
            set(list(data.orbits['2020042'].keys()))
            )




