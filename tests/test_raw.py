"""Unittests for the functions in raw, using example datasets."""

import unittest
import pandas.testing as pt
import pandas as pd
from gnssmapper import raw
import gnssmapper.common.time as tm
import gnssmapper.common.constants as cn

# class TestReadCSV(unittest.TestCase):
#     def setUp(self):
#         self.filedir = "./tests/exampleFiles/"
#         self.filepath = self.filedir+"log.txt"

#     def test_read_gnsslogger(self) -> None:
#         log = raw.read_gnsslogger(self.filepath)


#     def test_read_csv_(self) -> None:
#         raw_var,fix = raw.read_csv_(self.filepath)
#         raw_expected = pd.DataFrame()
#         fix_expected = pd.DataFrame()

#     def test_platform(self) -> None:
#         with open(filepath, 'r') as f:
#             wrong_platform = f.read()

#         wrong_platform[] = "G" #check the
#         self.assertWarnsRegex(UserWarning,"Platform not found in log file",raw.read_csv_,wrong_platform)

#     def test_version(self) -> None:
#         with open(filepath, 'r') as f:
#             wrong_version = f.read()

#         wrong_version[] = "G" #check the
#         self.assertRaisesRegex(ValueError,"found in log file. Gnssmapper supports gnsslogger v",raw.read_csv_,wrong_version)


#     def test_compare_version(self) -> None:
#         low = "1.3"
#         high = "1.4"
#         expected = "1.4.0.0"
#         self.assertTrue(raw._compare_version(high, expected))
#         self.assertFalse(raw._compare_version(low, expected))

class TestProcessRaw(unittest.TestCase):
    def setUp(self):
        # received 0.1 second after start of week
        rx_ = pd.DataFrame(
            {'week': [2000], 'day': [0], 'time': [0.1 * 10 ** 9]})
        # transmitted at 0.01 second after start of week
        tx_ = pd.DataFrame(
            {'week': [2000], 'day': [0], 'time': [0.01 * 10 ** 9]})

        d = {'Constellation': [1],
             'Svid': [1],
             'TimeNanos': tm.gpsweek_to_gps(rx_.week, rx_.day, rx_.time),
             'FullBiasNanos': [0],
             'ReceivedSvTimeNanos': tm.gpsweek_to_gps(0, tx_.day, tx_.time),
             'state': [9]}
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
        rx = self.input.TimeNanos[0]
        state = 9
        constellation = 'G'
        expected = tm.gpsweek_to_gps(self.rx.week, 0, 0)[0]
        self.assertEqual(
            raw.period_start_time(rx, state, constellation),
            expected)

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
        a=tm.gpsweek_to_gps(self.rx.week,self.rx.day,self.rx.time)
        b=self.tx_gps
        print(tm.gps_to_gpsweek(a))
        print(tm.gps_to_gpsweek(b))
        print(a - b)
        print(output.rx - output.tx)
        print (output.pr)
        pt.assert_series_equal(
            output.pr,
            pd.Series([0.09*cn.lightspeed]),
            check_names=False,check_exact=True)

    def test_tx_prior_week(self) -> None:
        # transmitted at 0.01 second before start of week
        tx_ = pd.DataFrame({'week': [1999], 'day': [6], 'time': [
            cn.nanos_in_day*7 - 0.01 * 10 ** 9]})
        tx_gps = tm.gpsweek_to_gps(tx_.week, tx_.day, tx_.time)
        input_ = self.input
        input_.loc[:,'ReceivedSvTimeNanos'] = tm.gpsweek_to_gps(0,tx_.day,tx_.time)
        output = raw.process_raw(input_)
        pt.assert_series_equal(
            output.tx,
            tx_gps,
            check_names=False,check_exact=True)
