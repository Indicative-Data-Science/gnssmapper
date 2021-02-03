""" Module containing functions to convert between time formats """
import pandas as pd
import constants


def gps_to_utc(time: pd.Series) -> pd.Series:
    """Converts nanoseconds since gps epoch into utc datetime format."""
    ts_gps = pd.to_datetime(time, units='ns', origin=constants.gps_epoch)
    # iterates leapsecond calculation to avoid problems at rollover.
    ts_estimate = ts_gps - ts_gps.map(constants.leap_seconds)

    return ts_gps - ts_estimate.map(constants.leap_seconds)


def utc_to_gps(time: pd.Series) -> pd.Series:
    """Converts utc datetime into nanoseconds since gps epoch."""
    return time.ts.delta + time.map(constants.leap_seconds) - constants.gps_epoch.delta


def gps_to_doy(time: pd.Series) -> pd.DataFrame:
    """Turn nanoseconds from gps epoch into a (YYYYDOY, nanoseconds) format."""
    ts = pd.to_datetime(time, units='ns', origin=constants.gps_epoch)

    year = ts.dt.year
    day = ts.dt.dayofyear.astype("str")
    yeardoy = year.str.cat(day.str.pad(3, side='left', fillchar="0"))

    ns = ts.dt.time.dt.delta

    return pd.Dataframe({'date':yeardoy,'time':ns})

def doy_to_gps(date:pd.Series,time: pd.Series) -> pd.Series:
    """Turn (YYYYDOY, nanoseconds) format into nanoseconds from gps epoch"""
    dt = pd.to_datetime(date,format=%Y%j)
    return dt.ts.delta + time - constants.gps_epoch.delta 

def gps_to_gpsweek(time: pd.Series) -> pd.DataFrame:
    """Turn nanoseconds from gps epoch into a (gpsweek,gpsday, nanoseconds) format."""
    days = time // constants.nanos_in_day
    gpsweek = days // 7
    gpsdays = tdiff.days - 7 * gpsweek
    ns = time - days * constants.nanos_in_day
    return pd.Dataframe({"week":gpsweek,"day":gpsdays,"time":ns})

def gpsweek_to_gps(week:pd.Series,day:pd.Series,time: pd.Series) -> pd.Series:
    """Turn (gpsweek,gpsday, nanoseconds) format into nanoseconds from gps epoch"""
    
    return ( week * 7 + day) * constants.nanos_in_day + time

