"""Unittests for the functions in time."""

import unittest
import pandas as pd
from gnssmapper.common.time import *
from gnssmapper.common.constants import gps_epoch
import pandas.testing as pt

class TestTime(unittest.TestCase):
    def test_gps_utc(self) -> None:
        ns = pd.Series([(1167264018 * 10**9),1167264018*10**9+1])
        ts = pd.Series([pd.Timestamp(year=2017,month=1,day=1,hour=0,minute=0,second=0,nanosecond=0),pd.Timestamp(year=2017,month=1,day=1,hour=0,minute=0,second=0,nanosecond=1)])
        pt.assert_series_equal(gps_to_utc(ns),ts,check_exact=True)
        pt.assert_series_equal(utc_to_gps(ts),ns,check_exact=True)
  
    def test_gps_doy(self) -> None:
        ns = pd.Series([100,200])
        ts = pd.DataFrame({'date': ['1980006', '1980006'], 'time': [100, 200]})
        print(gps_to_doy(ns))
        pt.assert_frame_equal(gps_to_doy(ns),ts,check_exact=True)
        pt.assert_series_equal(doy_to_gps(ts.date,ts.time),ns,check_exact=True)
    
    def test_gps_gpsweek(self) -> None:
        ns = pd.Series([604800 * 10**9])
        ts = pd.DataFrame({'week':[1],'day':[0],'time':[0]})
        pt.assert_frame_equal(gps_to_gpsweek(ns),ts,check_exact=True)
        pt.assert_series_equal(gpsweek_to_gps(ts.week,ts.day,ts.time),ns,check_exact=True,check_names=False)






