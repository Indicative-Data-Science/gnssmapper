"""
Module contains methods for processing raw GNSS data eg from gnss logger.
"""

import warnings
from typing import Tuple
from typing import Union
from io import StringIO

import pandas as pd
import numpy as np
import geopandas as gpd

import gnssmapper.common.constants as con
import gnssmapper.common.time as tm
import gnssmapper.common.check as check


def read_gnsslogger(filepath: str) -> gpd.GeoDataFrame:
    """processs a log file and returns a set of gnss receiverpoints.

    Parameters
    ----------
    filepath : str
        filepath (with filepath) of the log file.

    Returns
    -------
    gpd.DataFrame
        geopandas dataframe of gnss receiverpoints including:
            receiver position (as point geometry)
            time
            svid
            signal features e.g. CNO, pr

    """
    gnss_raw, gnss_fix = read_csv_(filepath)

    gnss_obs = process_raw(gnss_raw)
    points = join_receiver_position(
        gnss_obs, gnss_fix)

    check.receiverpoints(points)
    return points


def read_csv_(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the log file created by Google's gnsslogger Android app.

    Compatible with gnss logger version 1.4.0.0.

    Parameters
    ----------
    filepath : str
        filepath of the log file.

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame]:
        gnss_raw,gnss_fix

        gnss_raw
            all Raw (GnssClock and GnssMeasurement) observations from log file

        gnss_fix
            all Fix (position estimate) observations from log file, including:
                Latitude,Longitude,Altitude,(UTC)TimeInMs
    """

    version = ""
    platform = ""

    with open(filepath, 'r') as f:
        for line in f:
            if "version" in line.lower():
                start = line.lower().find("version")+9
                string = line[start:].split(" ", maxsplit=1)[0]
                version = ".".join(filter(str.isdigit,string))
                
            if "platform" in line.lower():
                start = line.lower().find("platform")+10
                platform = line[start:].split(" ", maxsplit=1)[0]

            if version != "" and platform != "":
                break

    if not _compare_platform(platform, con.minimum_platform):
        warnings.warn(
            f'Platform {platform} found in log file. Expected "Platform: N onwards".')

    if not _compare_version(version, con.minimum_version):
        raise ValueError(
            f'''Version {version} found in log file. Gnssmapper supports
            gnsslogger v{con.minimum_version} onwards''')

    with open(filepath, 'r') as f:
        raw = (line.split(",", maxsplit=1)[1].replace(
            " ", "") for line in f if "raw" in line.lower())
        gnss_raw = pd.read_csv(
            StringIO("\n".join(raw))
            ).convert_dtypes()

    with open(filepath, 'r') as f:
        fix = (line.split(",", maxsplit=1)[1].replace(
            " ", "") for line in f if "fix" in line.lower())
        gnss_fix = pd.read_csv(
            StringIO("\n".join(fix))
            ).convert_dtypes()

    return (gnss_raw, gnss_fix)


def _compare_version(actual: str, expected: str) -> bool:
    """Tests whether the log file version meets dependencies."""
    def str_to_list(x):
        a = [int(n) for n in x.split('.')]
        a += [0] * (4 - len(a))
        return a

    for a, e in zip(str_to_list(actual), str_to_list(expected)):
        if a > e:
            return True
        if a < e:
            return False

    return True

def _compare_platform(actual: Union[int,str], expected: int) -> bool:
    """Tests whether the OS version meets dependencies."""
    if str(actual).isdigit():
        return int(actual) >= int(expected)
    
    if actual in con.platform:
        return con.platform[actual] >= int(expected)
    
    else:
        warnings.warn(f'Platform {actual} found in log file, does not correspond to known platform number.')
        return False

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
    constellation = gnss_raw['ConstellationType'].map(
        con.constellation_numbering).convert_dtypes()
    svid_string = gnss_raw['Svid'].astype("string").str.pad(2, fillchar='0').convert_dtypes()
    svid = constellation.str.cat(svid_string)

    # compute receiver time (nanos since gps epoch)
    # gnss_raw['BiasNanos','TimeOffsetNanos] can provide subnanosecond accuracy
    rx = gnss_raw['TimeNanos'] - gnss_raw['FullBiasNanos']

    # compute transmission time (nanos since gps epoch)
    # ReceivedSvTimeNanos (time since start of gnss period)
    # + gps time of start of gnss period
    tx = (gnss_raw['ReceivedSvTimeNanos']
         - galileo_ambiguity(gnss_raw['ReceivedSvTimeNanos']).where(constellation == 'E', 0)
          + period_start_time(
              rx, gnss_raw['State'], constellation)
          + constellation.map(con.constellation_epoch_offset).convert_dtypes())

    # This will fail if the rx and tx are in seperate weeks
    # add code to remove a week if failed
    check = rx > tx
    if not check.all():
        warnings.warn(
            "rx less than tx, corrected assuming due to different gps weeks")
        correction = constellation.map(con.nanos_in_period).convert_dtypes()
        tx = tx.where(check,tx-correction)

    # check we have no nonsense psuedoranges
    assert 0 < np.min(rx - tx) <= np.max(rx -
                                         tx) < 1e9, f'Calculated pr time outside 0 to 1 seconds: {np.min(rx-tx)} - {np.max(rx-tx)}'

    # Pseudorange
    pr = (rx-tx) * (10**-9) * con.lightspeed

    # utc time
    time = tm.gps_to_utc(rx)
    time_ms = tm.utc_to_int(time) // 10**6

    return gnss_raw.assign(svid=svid, rx=rx, tx=tx, time=time, time_ms=time_ms, pr=pr)


def galileo_ambiguity(x: np.array) -> np.array:
    """ Correcting transmission time for Galileo measurements.

    Measurements may collapse into measuring pilot stage.
    100ms period is assumed to avoid ambiguity.
    """
    return (con.nanos_in_period['E']
            * (x // con.nanos_in_period['E']))


def period_start_time(rx: pd.Series, state: pd.Series, constellation: pd.Series) -> pd.Series:
    """Calculates the start time for the gnss period"""
    required_states = constellation.map(con.required_states)
    nanos_in_period = constellation.map(con.nanos_in_period).convert_dtypes()
    missing = required_states.isna() | state.isna()

    tx_valid = [not(m) and all(s & n for n in r) for s,r,m in zip(state,required_states,missing)]
    start = nanos_in_period * (rx // nanos_in_period)
    return start.where(tx_valid,pd.NA)
 


def join_receiver_position(
        gnss_obs: pd.DataFrame,
        gnss_fix: pd.DataFrame) -> gpd.GeoDataFrame:
    """  Add receiver positions to Raw data.

    Joined by utc time in milliseconds.
    """
    df = gnss_obs.join(gnss_fix.set_index("(UTC)TimeInMs"),
                       on="time_ms", how="inner",lsuffix="obs",rsuffix="fix")
    if len(df) != len(gnss_obs):
        warnings.warn(
            f'{len(gnss_obs)-len(df)} observations discarded without matching fix.'
        )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Latitude"], df["Longitude"],
                                    df["Altitude"]),
        crs=con.epsg_gnss_logger)
    return gdf
