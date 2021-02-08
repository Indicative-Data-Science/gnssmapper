"""Unittests for the functions in raw, using example datasets."""

import unittest
import pandas.testing as pt
import pandas as pd
import geopandas as gpd
from pygeos import Geometry
from gnssmapper import raw


class TestReadCSV(unittest.TestCase):
    def setUp(self):
        self.filedir = "./tests/exampleFiles/"
        self.filepath = self.filedir+"log.txt"       

    def test_read_gnsslogger(self) -> None:
        log = raw.read_gnsslogger(self.filepath)
        
        
    def test_read_csv_(self) -> None:
        raw_var,fix = raw.read_csv_(self.filepath)
        raw_expected = pd.DataFrame()
        fix_expected = pd.DataFrame()

    def test_platform(self) -> None:
        with open(filepath, 'r') as f:
            wrong_platform = f.read()

        wrong_platform[] = "G" #check the 
        self.assertWarnsRegex(UserWarning,"Platform not found in log file",raw.read_csv_,wrong_platform)
    
    def test_version(self) -> None:
        with open(filepath, 'r') as f:
            wrong_version = f.read()

        wrong_version[] = "G" #check the 
        self.assertRaisesRegex(ValueError,"found in log file. Gnssmapper supports gnsslogger v",raw.read_csv_,wrong_version)


    def test_compare_version(self) -> None:
        low = "1.3"
        high = "1.4"
        expected = "1.4.0.0"
        self.assertTrue(raw._compare_version(high, expected))
        self.assertFalse(raw._compare_version(low, expected))
    
    class TestProcessRAw(unittest.TestCase):
    def setUp(self):
        self.filedir = "./tests/exampleFiles/"
        self.filepath = self.filedir+"log.txt"       

    def test_read_gnsslogger(self) -> None:
        log = raw.read_gnsslogger(self.filepath)
        
    def test_process_raw(self) -> None:
        self.assertIsNone(rays(self.rays))
    
    def test_obs(self) -> None:
        valid_obs = gpd.GeoDataFrame(self.d, geometry=self.rays)
        self.assertIsNone(observations(valid_obs))

    def test_receiverpoints(self) -> None:
        valid_receiverpoints = gpd.GeoDataFrame(self.d, geometry = self.points)
        self.assertIsNone(receiverpoints(valid_receiverpoints))



