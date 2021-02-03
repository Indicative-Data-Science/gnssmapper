"""
Module contains methods for processing raw GNSS data eg from gnss logger.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from math import floor
import constants
import gpstime
import warnings
from typing import Tuple


def read_gnsslogger(filename: str) -> gpd.GeoDataFrame:
    """processs a gnsslogger log file (csv) and returns a set of gnss receiverpoints.

    Parameters
    ----------
    filename : str
        filename (with filepath) of the log file.

    Returns
    -------
    gpd.DataFrame
        geopandas dataframe of gnss receiverpoints including:
            receiver position (as point geometry)
            rx, time, time_ms (observation time in nanos since gps epoch, in utc format, and millis since utc epoch)
            svid: svid identifier in IGS format
            signal features e.g. CNO, pr

    """
    gnss_raw, gnss_fix = read_csv_(filename)
    gnss_obs = process_raw(gnss_raw)
    gnss_obs = join_receiver_position(
        gnss_obs, gnss_fix)
    return gnss_obs


def read_csv_(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the log file created by Google's gnsslogger Android app.

    Compatible with gnss logger version 2.0.

    Parameters
    ----------
    filename : str
        filename (with filepath) of the log file.

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame]:
        gnss_raw,gnss_fix

        gnss_raw
            all Raw (GnssClock and GnssMeasurement) observations from log file

        gnss_fix
            all Fix (position solutions from Android phone) observations from log file, including:
                Latitude,Longitude,Altitude,time Time and time nanos
    """

    version = ""
    platform = ""

    with open(filename, 'r') as f:
        for line in f:
            if "version" in line.lower():
                start = line.lower().find("version")+9
                end = start+8
                version = line[start:].split(" ", maxsplit=1)[0]
            if "platform" in line.lower():
                start = line.lower().find("platform")+10
                end = start+2
                platform = line[start:end]

            if version != "" and platform != "":
                break

    if platform != "N":
        warnings.warn(
            'Platform not found in log file. Expected "Platform: N".')

    if not _compare_version(version, constants.minimum_version):
        raise ValueError(
            f'Version {version} found in log file. Gnssmapper supports gnsslogger v{constants.minimum_version} onwards')

    with open(filename, 'r') as f:
        raw = (l.split(",", maxsplit=1)[1].replace(
            " ", "") for l in f if "raw" in l.lower())
        gnss_raw = pd.read_csv("\n".join(raw))

    with open(filename, 'r') as f:
        fix = (l.split(",", maxsplit=1)[1].replace(
            " ", "") for l in f if "fix" in l.lower())
        gnss_fix = pd.read_csv("\n".join(fix))

    return (gnss_raw, gnss_fix)


def _compare_version(actual: str, expected: str) -> bool:
    """Tests whether the log file version meets dependencies."""
    def str_to_list(x):
        l = [int(n) for n in x.split('.')]
        l += [0] * (4 - len(l))
        return l

    for a, e in zip(str_to_list(actual), str_to_list(expected)):
        if a > e:
            return True
        if a < e:
            return False

    return True


def process_raw(gnss_raw: pd.DataFrame) -> pd.DataFrame:
    """Generates signal features from raw measurements.

    Parameters
    ----------
    gnss_raw : pd.DataFrame
        all Raw (GnssClock and GnssMeasurement) observations from log file

    Returns
    -------
    pd.DataFrame
        gnss_raw plus svid, rx, tx, time, pr
    """
    # reformat svid to standard (IGS) format
    constellation = gnss_raw['Constellation'].map(
        constants.constellation_numbering)
    svid_string = gnss_raw['Svid'].astype("string").pad(2, fillchar='0')
    svid = constellation.str.cat(svid_string)

    # compute rx (nanos since gps epoch)
    # gnss_raw['BiasNanos','TimeOffsetNanos] can provide subnanosecond accuracy
    rx = gnss_raw['TimeNanos'] - gnss_raw['FullBiasNanos']

    # compute transmission time (nanos since gps epoch)
    # ReceivedSvTimeNanos (time since start of gnss period) + gps time of start of gnss period
    tx = gnss_raw['ReceivedSvTimeNanos'] - np.where(
        constellation == 'E', galileo_ambiguity(gnss_raw['ReceivedSvTimeNanos']), 0)
    + np.vectorize(period_start_time)(rx, gnss_raw['state'], constellation)
    + constellation.map(constants.constellation_epoch_offset)

    # This will fail if the rx and tx are in seperate weeks
    # add code to remove a week if failed
    if np.any(rx < tx):
        warnings.warn(
            "rx less than tx, corrected assuming due to different gps weeks")
        tx -= np.where(rx < tx,
                       constellation.map(constants.nanos_in_period), 0)

    # check we have no nonsense psuedoranges
    assert 0 < rx - tx < 1e9, 'Calculated pr time outside 0 to 1 seconds'

    # Pseudorange
    pr = (rx-tx) * 1e-9 * constants.lightspeed

    # utc time
    time = time.gps_to_utc(rx)
    time_ms = time.astype(int) // 1e6

    return pd.concat([gnss_raw, svid, rx, tx, time, time_ms, pr])


def galileo_ambiguity(x: int) -> int:
    """ Galileo measurements may collapse into measuring pilot stage with 100ms ambiguity """
    return constants.nanos_in_period['E'] * np.floor(x / constants.nanos_in_period['E'])


def period_start_time(rx: int, state: int, constellation: str) -> int:
    """Calculates the start time for the gnss period (since gps epoch in nanoseconds)"""
    tx_valid = all(state & n for n in constants.required_states[constellation])
    if tx_valid:
        return constants.nanos_in_period[constellation] * floor(
            rx / constants.nanos_in_period[constellation])
    else:
        return np.nan


def join_receiver_position(gnss_obs: pd.DataFrame, gnss_fix: pd.DataFrame) ->gpd.GeoDataFrame:
    """ Add receiver positions to observation data, jonied by time time in Ms """
    df = gnss_obs.join(gnss_fix.set_index("(time)TimeInMS"),
                       on="time_ms", how="inner")
    if len(df) != len(gnss_obs):
        warnings.warn(
            f"{len(gnss_obs)-len(df)} observations discarded without matching fix."
        )
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
        df["Latitude"], df["Longitude"], df["Altitude"]), crs=constants.epsg_gnss_logger)
    return gdf
