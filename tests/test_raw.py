"""Unittests for the functions in raw, using example datasets."""

import unittest
import pandas.testing as pt
import pandas as pd
from io import StringIO

from gnssmapper import raw
import gnssmapper.common.time as tm
import gnssmapper.common.constants as cn

class TestReadCSV(unittest.TestCase):
    def setUp(self):
        self.filedir = "./tests/data/"
        self.filepath = self.filedir+"log_20200211.txt"

    def test_read_gnsslogger(self) -> None:
        log = raw.read_gnsslogger(self.filepath)


    def test_read_csv_(self) -> None:
        raw_var,fix = raw.read_csv_(self.filepath)
        raw_expected = pd.DataFrame()
        fix_expected = pd.DataFrame()

    def test_platform(self) -> None:
        #copy of log.txt with platform replaced by 6
        wrong_platform = self.filedir+"wrong_platform.txt"
        self.assertWarnsRegex(UserWarning,"Platform 6 found in log file",raw.read_csv_,wrong_platform)

    def test_version(self) -> None:
         #copy of log.txt with version replaced by 1.3.9.9
        wrong_version = self.filedir+"wrong_version.txt"
        self.assertRaisesRegex(ValueError,"Version 1.3.9.9 found in log file",raw.read_csv_,wrong_version)


    def test_compare_version(self) -> None:
        low = "1.3"
        high = "1.4"
        expected = "1.4.0.0"
        self.assertTrue(raw._compare_version(high, expected))
        self.assertFalse(raw._compare_version(low, expected))

    def test_compare_platform(self) -> None:
        low = set(["6","M",6])
        high = set(["7","N",7,"O",10,"10"])
        expected = "7"
        self.assertTrue(all([raw._compare_platform(x, expected) for x in high]))
        self.assertFalse(any([raw._compare_platform(x, expected) for x in low]))


class TestProcessRaw(unittest.TestCase):
    def setUp(self):
        # received 0.1 second after start of week
        rx_ = pd.DataFrame(
            {'week': [2000], 'day': [0], 'time': [1 * 10 ** 8]})
        # transmitted at 0.01 second after start of week
        tx_ = pd.DataFrame(
            {'week': [2000], 'day': [0], 'time': [1 * 10 ** 7]})

        d = {'ConstellationType': [1],
             'Svid': [1],
             'TimeNanos': tm.gpsweek_to_gps(rx_.week, rx_.day, rx_.time),
             'FullBiasNanos': [0],
             'ReceivedSvTimeNanos': tm.gpsweek_to_gps(0, tx_.day, tx_.time),
             'State': [9]}
        self.input = pd.DataFrame(d)
        self.tx_gps = tm.gpsweek_to_gps(tx_.week, tx_.day, tx_.time)
        self.rx = rx_

    def test_galileo_ambiguity(self) -> None:
        import numpy as np
        expected = np.array([6, 7, 8, 9, 10])*cn.nanos_in_period['E']
        testdata = np.array([1, 2, 3, 4, 5])+expected
        np.testing.assert_array_equal(
            raw.galileo_ambiguity(testdata),
            expected)

    def test_period_start_time(self) -> None:
        import numpy as np
        rx = self.input.TimeNanos[0]
        state = 9
        constellation = 'G'
        expected = tm.gpsweek_to_gps(pd.Series([self.rx.week]),
                                     pd.Series([0]),
                                     pd.Series([0]))
        np.testing.assert_array_equal(
            raw.period_start_time(rx, state, constellation),
            expected.to_list())

    def test_svid(self) -> None:
        output = raw.process_raw(self.input)
        pt.assert_series_equal(
            output.svid,
            pd.Series(["G01"], name='svid'),
            check_names=False)

    def test_rx(self) -> None:
        output = raw.process_raw(self.input)
        expected = tm.gpsweek_to_gps(self.rx.week,self.rx.day,self.rx.time)
        pt.assert_series_equal(
            output.rx,
            expected,
            check_names=False,check_exact=True)

    def test_tx(self) -> None:
        output = raw.process_raw(self.input)
        pt.assert_series_equal(
            output.tx,
            self.tx_gps,
            check_names=False,check_exact=True)

    def test_pr(self) -> None:
        output = raw.process_raw(self.input)
        pt.assert_series_equal(
            output.pr-pd.Series([9*cn.lightspeed/100]),
            pd.Series([0.0]),
            check_names=False)

    def test_tx_prior_week(self) -> None:
        # transmitted at 0.01 second before start of week
        tx_ = pd.DataFrame({'week': [1999], 'day': [6], 'time': [
            cn.nanos_in_day*7 - 1 * 10 ** 7]})
        tx_gps = tm.gpsweek_to_gps(tx_.week, tx_.day, tx_.time)
        input_ = self.input
        input_.loc[:,'ReceivedSvTimeNanos'] = tm.gpsweek_to_gps(0,tx_.day,tx_.time)
        output = raw.process_raw(input_)
        pt.assert_series_equal(
            output.tx,
            tx_gps,
            check_names=False,check_exact=True)

# class TestJoinReceiverPosition(unittest.TestCase):
    # def test_warning_if_dropped(self) -> None:

    # def test_joining(self) -> None:
