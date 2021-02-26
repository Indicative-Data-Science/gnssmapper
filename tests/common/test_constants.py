"""Unittests for the functions in constants."""

import unittest
import pandas as pd
from gnssmapper.common.constants import leap_seconds


class TestConstants(unittest.TestCase):
    def test_leapsecond(self) -> None:
        undefined = pd.Series(pd.Timestamp(year=2014,month=12,day=31,hour=23,minute=59,second=59))
        pre = pd.Series(pd.Timestamp(year=2016,month=12,day=31,hour=23,minute=59,second=59))
        post = pd.Series(pd.Timestamp(year=2017, month=1, day=1, hour=0, minute=0, second=0))
        missing = pd.Series(pd.NaT)
        pd.testing.assert_series_equal(leap_seconds(pre), pd.Series(17).convert_dtypes())
        pd.testing.assert_series_equal(leap_seconds(post), pd.Series(18).convert_dtypes())
        self.assertWarns(UserWarning, leap_seconds, undefined)
        self.assertTrue(pd.isnull(
                                 leap_seconds(undefined)
                                 ).all())
        self.assertTrue(pd.isnull(
                                 leap_seconds(missing)
                                 ).all())





