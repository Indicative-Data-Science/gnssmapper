"""Unittests for the functions in constants."""

import unittest
import pandas as pd
from gnssmapper.common.constants import leap_seconds


class TestConstants(unittest.TestCase):
    def test_leapsecond(self) -> None:
        undefined = pd.Timestamp(year=2014,month=12,day=31,hour=23,minute=59,second=59)
        pre = pd.Timestamp(year=2016,month=12,day=31,hour=23,minute=59,second=59)
        post = pd.Timestamp(year=2017,month=1,day=1,hour=0,minute=0,second=0)
        self.assertEqual(leap_seconds(pre), 17)
        self.assertEqual(leap_seconds(post), 18)
        self.assertRaises(ValueError,leap_seconds,undefined)





