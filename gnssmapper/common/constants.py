"""
Module containing constants and definitions used in gnss processing.

Constellation specific information stored as dictionaries.
keys [G,R,C,E] refer to gps,glonasss, beidou, and galileo constellations """

import bisect
import warnings

import pandas as pd
import numpy as np

""" GNSS logger"""

# minimum version of gnsslogger
minimum_version = "1.4.0.0"
minimum_platform = 7
platform = {"N": 7, "O": 8, "P": 9}

# supported constellations
supported_constellations = set(["G", "R", "C", "E"])

# IGS constellation identifiers
constellation_numbering = {1: "G", 3: "R", 5: "C", 6: "E"}

# navigation signal states indicating no measurement ambiguity
required_states = {
    'G': [1, 8],
    'R': [1, 8, 128],
    'C': [1, 8],
    'E': [1, 8, 2048],
}

epsg_gnss_logger = 'EPSG:4979'


""" GNSS constants """

# number of nanoseconds in a gnss period
nanos_in_period = {
    'G': 604800 * 10**9,
    'R': 86400 * 10**9,
    'C': 604800 * 10**9,
    'E': 10**8,
}

# offset of gnss epoch from gps
# constellation_epoch_offset = {
#     'G': 0,
#     'R': 86400 * 10**9,
#     'C': 14 * 10**9,
#     'E': -10800*10**9 + 18*10**9,
# }
constellation_epoch_offset = {
    'G': 0,
    'R': - 3* 3600 * 10**9,
    'C': 14 * 10**9,
    'E': 0,
}



# lightspeed in m/s
lightspeed = 299792458

""" GPS and UTC conversion"""
nanos_in_day = 86400 * 10**9

# starting epoch for gps
gps_epoch = pd.to_datetime('1980-01-06', format="%Y-%m-%d")


def leap_seconds(time:pd.Series) -> pd.Series:
    """gps leap seconds."""
    # add to lists as gps seconds announced
    ls_dates_str = ['2015-07-01', '2017-01-01']
    ls_dates = pd.to_datetime(ls_dates_str, format="%Y-%m-%d")
    ls = np.array([np.nan, 17, 18])
    idx = np.searchsorted(ls_dates,time,side='right')
    if np.min(idx) == 0:
        warnings.warn(f"GPS leap seconds only defined post {ls_dates[0]}")
    output = np.where(pd.isnull(time),np.nan,ls[idx])  

    return pd.Series(output,index=time.index,name=time.name).convert_dtypes()


epsg_satellites = 'EPSG:4978'
epsg_wgs84 = 'EPSG:4979'
epsg_wgs84_cart = 'EPSG:4978'
ray_length = 1000
minimum_elevation = 0
maximum_elevation = 85

