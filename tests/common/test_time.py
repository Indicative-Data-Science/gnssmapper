"""Unittests for the functions in time."""

import unittest
import numpy as np
import pandas as pd
import gnssmapper.common.time as tm
from gnssmapper.common.constants import gps_epoch
import pandas.testing as pt

class TestTime(unittest.TestCase):
    def test_gps_utc(self) -> None:
        ns = pd.Series([(1167264018 * 10**9),1167264018*10**9+1]).convert_dtypes()
        ts = pd.Series([pd.Timestamp(year=2017,month=1,day=1,hour=0,minute=0,second=0,nanosecond=0),pd.Timestamp(year=2017,month=1,day=1,hour=0,minute=0,second=0,nanosecond=1)])
        pt.assert_extension_array_equal(tm.gps_to_utc(ns).array,ts.array,check_exact=True)
        pt.assert_extension_array_equal(tm.utc_to_gps(ts).array,ns.array,check_exact=True)
  
    def test_gps_doy(self) -> None:
        ns = pd.Series([1,2]).convert_dtypes()
        ts = pd.DataFrame({'date': ['1980006', '1980006'], 'time': [1, 2]}).convert_dtypes()
        pt.assert_frame_equal(tm.gps_to_doy(ns).astype('float64'),ts.astype('float64'),check_exact=True,check_dtype=False)
        pt.assert_extension_array_equal(tm.doy_to_gps(ts.date,ts.time).array,ns.array,check_exact=True)
    
    def test_gps_gpsweek(self) -> None:
        ns = pd.Series([604800*2000 * 10**9 + 1 * 10 ** 7]).convert_dtypes()
        ts = pd.DataFrame({'week':[2000],'day':[0],'time':[1 * 10 ** 7]}).convert_dtypes()
        pt.assert_frame_equal(tm.gps_to_gpsweek(ns).astype('float64'),ts.astype('float64'),check_exact=True)
        pt.assert_extension_array_equal(tm.gpsweek_to_gps(ts.week,ts.day,ts.time).array,ns.array,check_exact=True)


class TestMissing(unittest.TestCase):
    def test_gps_utc(self) -> None:
        ns = pd.Series([(1167264018 * 10**9),pd.NA],dtype='Int64')
        ts = pd.Series([pd.Timestamp(year=2017, month=1, day=1, hour=0, minute=0, second=0, nanosecond=0), pd.NaT])
        pt.assert_extension_array_equal(tm.gps_to_utc(ns).array,ts.array,check_exact=True)
        pt.assert_extension_array_equal(tm.utc_to_gps(ts).array,ns.array,check_exact=True)
  
    def test_gps_doy(self) -> None:
        ns = pd.Series([1, 2, pd.NA], dtype='Int64')
        ts = pd.DataFrame({'date': ['1980006', '1980006',pd.NA], 'time': [1, 2,np.nan]}).convert_dtypes()
        pt.assert_frame_equal(tm.gps_to_doy(ns).astype('float64'),ts.astype('float64'),check_exact=True,check_dtype=False)
        pt.assert_extension_array_equal(tm.doy_to_gps(ts.date,ts.time).array,ns.array,check_exact=True)
    
    def test_gps_gpsweek(self) -> None:
        ns = pd.Series([604800*2000 * 10**9 + 1 * 10 ** 7,pd.NA],dtype='Int64')
        ts = pd.DataFrame({'week':[2000,pd.NA],'day':[0,pd.NA],'time':[1 * 10 ** 7,pd.NA]},dtype='Int64')
        pt.assert_frame_equal(tm.gps_to_gpsweek(ns).astype('float64'),ts.astype('float64'),check_exact=True)
        pt.assert_extension_array_equal(tm.gpsweek_to_gps(ts.week,ts.day,ts.time).array,ns.array,check_exact=True)


class TestHelper(unittest.TestCase):

    def test_nanos(self) -> None:
        d=pd.Series([1.0])
        self.assertWarns(UserWarning,tm._check_nanos,d)

class TestIndex(unittest.TestCase):
    def test_gps_utc(self) -> None:
        ns = pd.Series([(1167264018 * 10**9),1167264018*10**9+1],index=[2,3],name='foo').convert_dtypes()
        ts = pd.Series([pd.Timestamp(year=2017,month=1,day=1,hour=0,minute=0,second=0,nanosecond=0),pd.Timestamp(year=2017,month=1,day=1,hour=0,minute=0,second=0,nanosecond=1)],index=[2,3],name='foo').convert_dtypes()
        pt.assert_series_equal(tm.gps_to_utc(ns),ts,check_exact=True)
        pt.assert_series_equal(tm.utc_to_gps(ts),ns,check_exact=True)
  
    def test_gps_doy(self) -> None:
        ns = pd.Series([1,2],index=[2,3],name='foo').convert_dtypes()
        ts = pd.DataFrame({'date': ['1980006', '1980006'], 'time': [1, 2]},index=[2,3]).convert_dtypes()
        pt.assert_frame_equal(tm.gps_to_doy(ns),ts,check_exact=True)
        pt.assert_series_equal(tm.doy_to_gps(ts.date,ts.time),ns,check_names=False, check_exact=True)
    
    def test_gps_gpsweek(self) -> None:
        ns = pd.Series([604800*2000 * 10**9 + 1 * 10 ** 7],index=[2],name='foo').convert_dtypes()
        ts = pd.DataFrame({'week':[2000],'day':[0],'time':[1 * 10 ** 7]},index=[2]).convert_dtypes()
        pt.assert_frame_equal(tm.gps_to_gpsweek(ns),ts,check_exact=True)
        pt.assert_series_equal(tm.gpsweek_to_gps(ts.week,ts.day,ts.time),ns,check_exact=True,check_names=False)

