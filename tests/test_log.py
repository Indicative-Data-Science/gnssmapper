"""Unittests for the functions in raw, using example datasets."""

import unittest
import pandas.testing as pt
import pandas as pd
from io import StringIO

from gnssmapper import log
import gnssmapper.common.time as tm
import gnssmapper.common.constants as cn

class TestReadCSV(unittest.TestCase):
    def setUp(self):
        self.filedir = "./tests/data/"
        self.filepath = self.filedir+"log_20200211.txt"

    def test_read_csv_(self) -> None:
        raw_var,fix = log.read_csv_(self.filepath)
        raw_expected = pd.DataFrame({
            'TimeNanos': [34554000000],
            'FullBiasNanos':[-1265446151445559028],
            'Svid': [2],
            'ConstellationType': [1],
            'State': [16431],
            'Cn0DbHz': [22.340620040893555]}).convert_dtypes()
        fix_expected = pd.DataFrame({
            'Latitude': [51.524707],
            'Longitude': [-0.134140],
            'Altitude': [114.858938],
            '(UTC)TimeInMs': [1581410967999]
        }).convert_dtypes()
        pt.assert_frame_equal(
            raw_var.loc[0:0, ['TimeNanos','FullBiasNanos','Svid', 'ConstellationType','State','Cn0DbHz']],
            raw_expected)
        pt.assert_frame_equal(
            fix.loc[0:0, ['Latitude', 'Longitude', 'Altitude', '(UTC)TimeInMs']],
            fix_expected
        )


    def test_platform(self) -> None:
        #copy of log.txt with platform replaced by 6
        wrong_platform = self.filedir+"wrong_platform.txt"
        self.assertWarnsRegex(UserWarning,"Platform 6 found in log file",log.read_csv_,wrong_platform)

    def test_version(self) -> None:
         #copy of log.txt with version replaced by 1.3.9.9
        wrong_version = self.filedir+"wrong_version.txt"
        self.assertRaisesRegex(ValueError,"Version 1.3.9.9 found in log file",log.read_csv_,wrong_version)


    def test_compare_version(self) -> None:
        low = "1.3"
        high = "1.4"
        expected = "1.4.0.0"
        self.assertTrue(log._compare_version(high, expected))
        self.assertFalse(log._compare_version(low, expected))

    def test_compare_platform(self) -> None:
        low = set(["6","M",6])
        high = set(["7","N",7,"O",10,"10"])
        expected = "7"
        self.assertTrue(all([log._compare_platform(x, expected) for x in high]))
        self.assertFalse(any([log._compare_platform(x, expected) for x in low]))

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
        self.tx_gps = tm.gpsweek_to_gps(tx_.week, tx_.day, tx_.time).convert_dtypes()
        self.rx = rx_

    def test_galileo_ambiguity(self) -> None:
        import numpy as np
        expected = np.array([6, 7, 8, 9, 10])*cn.nanos_in_period['E']
        testdata = np.array([1, 2, 3, 4, 5])+expected
        np.testing.assert_array_equal(
            log.galileo_ambiguity(testdata),
            expected)

    def test_period_start_time(self) -> None:
        import numpy as np
        rx = self.input.TimeNanos[0]
        state = 9
        constellation = 'G'
        expected = tm.gpsweek_to_gps(self.rx.week,
                                     pd.Series([0]),
                                     pd.Series([0]))
        pt.assert_series_equal(
            log.period_start_time(pd.Series([rx]), pd.Series([state]), pd.Series([constellation])),
            expected,check_names=False,check_dtype=False)

    def test_svid(self) -> None:
        output = log.process_raw(self.input)
        pt.assert_series_equal(
            output.svid,
            pd.Series(["G01"], name='svid').convert_dtypes(),
            check_names=False)

    def test_rx(self) -> None:
        output = log.process_raw(self.input)
        expected = tm.gpsweek_to_gps(self.rx.week,self.rx.day,self.rx.time)
        pt.assert_extension_array_equal(
            output.rx.array,
            expected.array,
            check_exact=True)

    def test_tx(self) -> None:
        output = log.process_raw(self.input)
        pt.assert_extension_array_equal(
            output.tx.array,
            self.tx_gps.array,
            check_exact=True)

    def test_pr(self) -> None:
        output = log.process_raw(self.input)
        pt.assert_series_equal(
            output.pr-pd.Series([9*cn.lightspeed/100]),
            pd.Series([0.0]).convert_dtypes(convert_integer=False),
            check_names=False)

    def test_tx_prior_week(self) -> None:
        # transmitted at 0.01 second before start of week
        tx_ = pd.DataFrame({'week': [1999], 'day': [6], 'time': [
            cn.nanos_in_day - 1 * 10 ** 7]})
        tx_gps = tm.gpsweek_to_gps(tx_.week, tx_.day, tx_.time).convert_dtypes()
        input_ = self.input
        input_.loc[:,'ReceivedSvTimeNanos'] = tm.gpsweek_to_gps(0,0,tx_.day*cn.nanos_in_day+tx_.time)
        output = log.process_raw(input_)
        pt.assert_extension_array_equal(
            output.tx.array,
            tx_gps.array,
            check_exact=True)

class TestNA(unittest.TestCase):
    def setUp(self):
        self.filedir = "./tests/data/"
        self.filepath = self.filedir+"missing.txt"

    def test_datatype(self) -> None:
        raw_var, fix = log.read_csv_(self.filepath)
        test_columns = ["TimeNanos", "FullBiasNanos", 'ReceivedSvTimeNanos', 'ConstellationType', 'Svid', 'State']
        self.assertTrue(all(raw_var[c].dtype=='Int64' for c in test_columns))
        self.assertTrue(fix['(UTC)TimeInMs'].dtype=='Int64')

    def test_processing(self) -> None:
        raw_var, _ = log.read_csv_(self.filepath)
        log.process_raw(raw_var)
    
    def test_joining(self) -> None:
        raw_var, gnss_fix = log.read_csv_(self.filepath)
        gnss_obs = log.process_raw(raw_var)
        self.assertWarnsRegex(UserWarning,
                              '37 observations discarded without matching fix.',log.join_receiver_position,
            gnss_obs, gnss_fix)

class TestJoinReceiverPosition(unittest.TestCase):
    def setUp(self):
        self.filedir = "./tests/data/"
        self.filepath = self.filedir+"log_20200211.txt"
        raw_var, fix = log.read_csv_(self.filepath)
        self.gnss_obs = log.process_raw(raw_var[0:50])
        self.gnss_fix=fix[0:1]
    
    def test_joining(self) -> None:
        epoch = self.gnss_obs[0:35]
        result = log.join_receiver_position(epoch, self.gnss_fix)
        self.assertEqual(len(result), 35)
        self.assertEqual(result.time.nunique(), 1)
        self.assertEqual(result.Longitude.nunique(), 1)
    
    def test_warning_on_drop(self) -> None:
        self.assertWarnsRegex(UserWarning,
                              '15 observations discarded without matching fix.',
                              log.join_receiver_position,
                              self.gnss_obs, self.gnss_fix)