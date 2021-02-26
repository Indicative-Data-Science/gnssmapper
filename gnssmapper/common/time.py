""" Module containing functions to convert between time formats """
import warnings

import pandas as pd
import numpy as np

from gnssmapper.common.constants import gps_epoch,leap_seconds,nanos_in_day

def gps_to_utc(time: pd.Series) -> pd.Series:
    """Converts nanos since gps epoch into utc datetime format."""
    _check_nanos(time)
    time_int = time.astype('Int64')
    #origin defined indirectly due to incompatibility with Int64 
    ts_gps = pd.to_datetime(time_int+gps_epoch.value, unit='ns')
    # iterates leapsecond calculation to avoid problems at rollover.
    
    ts_estimate = ts_gps - pd.to_timedelta(leap_seconds(ts_gps),unit='s')
    return ts_gps - pd.to_timedelta(leap_seconds(ts_estimate),unit='s')


def utc_to_gps(time: pd.Series) -> pd.Series:
    """Converts utc datetime into nanos since gps epoch."""
    delta = (time
            + pd.to_timedelta(leap_seconds(time),unit='s',errors='coerce')
            - gps_epoch)
    return utc_to_int(delta)



def gps_to_doy(time: pd.Series) -> pd.DataFrame:
    """Turn nanos from gps epoch into a (YYYYDOY, nanos) format."""
    _check_nanos(time)
    time_int = time.astype('Int64')
    ts = pd.to_datetime(time_int + gps_epoch.value, unit='ns')
    
    year = ts.dt.year.astype('Int64').astype("string")
    day = ts.dt.dayofyear.astype('Int64').astype("string")
    yeardoy = year.str.cat(day.str.pad(3, side='left', fillchar="0"))
    
    time_in_day = ts - pd.to_datetime(yeardoy, format='%Y%j',errors='coerce')
    ns = utc_to_int(time_in_day)
    return pd.DataFrame({'date': yeardoy, 'time': ns})


def doy_to_gps(date: pd.Series, time: pd.Series) -> pd.Series:
    """Turn (YYYYDOY, nanos) format into nanos from gps epoch"""
    _check_nanos(time)
    time_int = time.astype('Int64')
    dt = pd.to_datetime(date, format='%Y%j',errors='coerce')
    delta = dt + pd.to_timedelta(time_int, unit='ns') - gps_epoch

    return utc_to_int(delta)


def gps_to_gpsweek(time: pd.Series) -> pd.DataFrame:
    """Turn nanos from gps epoch into a (gpsweek,gpsday, ns) format."""
    _check_nanos(time)
    time_int=time.astype('Int64')
    days = time_int.floordiv(nanos_in_day)
    gpsweek = days.floordiv(7)
    gpsdays = days - 7 * gpsweek
    ns = time - days * nanos_in_day
    return pd.DataFrame({"week": gpsweek, "day": gpsdays, "time": ns})


def gpsweek_to_gps(
        week: pd.Series, day: pd.Series,
        time: pd.Series) -> pd.Series:
    """Turn (gpsweek,gpsday, nanos) format into nanos from gps epoch"""
    _check_nanos(time)
    time_int=time.astype('Int64')
    return pd.Series((week * 7 + day) * nanos_in_day + time_int,name="time")

def utc_to_int(time: pd.Series) -> pd.Series:
    """ Turns a timestamp into number of nanoseconds (propogates NaT values) """
    return pd.Series(time.array, dtype='Int64',index=time.index,name=time.name).where(time.notna(), pd.NA).astype('Int64')
    

def _check_nanos(time: pd.Series) -> None:
    """ Warns if floats being used in for gps time"""
    if pd.api.types.is_float_dtype(time):
        warnings.warn("Potential rounding errors due to GPS time in nanoseconds input as float")
    return None