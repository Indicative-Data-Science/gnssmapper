""" Module containing functions to convert between time formats """
import pandas as pd

from . import constants


def gps_to_utc(time: pd.Series) -> pd.Series:
    """Converts nanos since gps epoch into utc datetime format."""
    ts_gps = pd.to_datetime(time, unit='ns', origin=constants.gps_epoch)
    # iterates leapsecond calculation to avoid problems at rollover.
    ts_estimate = ts_gps - pd.to_timedelta(ts_gps.map(constants.leap_seconds),unit='s')
    return ts_gps - pd.to_timedelta(ts_estimate.map(constants.leap_seconds),unit='s')


def utc_to_gps(time: pd.Series) -> pd.Series:
    """Converts utc datetime into nanos since gps epoch."""
    delta = (time
            + pd.to_timedelta(time.map(constants.leap_seconds),unit='s')
            - constants.gps_epoch) 
    return pd.Series(delta.values,dtype='int64')


def gps_to_doy(time: pd.Series) -> pd.DataFrame:
    """Turn nanos from gps epoch into a (YYYYDOY, nanos) format."""
    ts = pd.to_datetime(time, unit='ns', origin=constants.gps_epoch)

    year = ts.dt.year.astype("str")
    day = ts.dt.dayofyear.astype("str")
    yeardoy = year.str.cat(day.str.pad(3, side='left', fillchar="0"))
    time_in_day = ts - pd.to_datetime(yeardoy, format='%Y%j')
    ns = pd.Series(time_in_day.values,dtype='int64')

    return pd.DataFrame({'date': yeardoy, 'time': ns})


def doy_to_gps(date: pd.Series, time: pd.Series) -> pd.Series:
    """Turn (YYYYDOY, nanos) format into nanos from gps epoch"""
    dt = pd.to_datetime(date, format='%Y%j')
    delta = dt + pd.to_timedelta(time,unit='ns') - constants.gps_epoch
    return pd.Series(delta.values,dtype='int64')


def gps_to_gpsweek(time: pd.Series) -> pd.DataFrame:
    """Turn nanos from gps epoch into a (gpsweek,gpsday, ns) format."""
    days = time // constants.nanos_in_day
    gpsweek = days // 7
    gpsdays = days - 7 * gpsweek
    ns = time - days * constants.nanos_in_day
    return pd.DataFrame({"week": gpsweek, "day": gpsdays, "time": ns})


def gpsweek_to_gps(
        week: pd.Series, day: pd.Series,
        time: pd.Series) -> pd.Series:
    """Turn (gpsweek,gpsday, nanos) format into nanos from gps epoch"""

    return pd.Series((week * 7 + day) * constants.nanos_in_day + time,name="time")
