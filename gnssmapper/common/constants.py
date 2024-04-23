"""
Module containing constants and definitions used in gnss processing.

Constellation specific information stored as dictionaries.
keys [G,R,C,E] refer to gps,glonasss, beidou, and galileo constellations """

import warnings

import numpy as np
import pandas as pd

""" GNSS logger"""

# minimum version of gnsslogger
minimum_version = "1.4.0.0"
minimum_platform = 7
platform = {"N": 7, "O": 8, "P": 9}

# supported constellations
supported_constellations = {"G", "R", "C", "E"}
supported_svids = {
    "G": {
        "G01",
        "G02",
        "G03",
        "G04",
        "G05",
        "G06",
        "G07",
        "G08",
        "G09",
        "G10",
        "G11",
        "G12",
        "G13",
        "G14",
        "G15",
        "G16",
        "G17",
        "G18",
        "G19",
        "G20",
        "G21",
        "G22",
        "G23",
        "G24",
        "G25",
        "G26",
        "G27",
        "G28",
        "G29",
        "G30",
        "G31",
        "G32",
    },
    "R": {
        "R01",
        "R02",
        "R03",
        "R04",
        "R05",
        "R06",
        "R07",
        "R08",
        "R09",
        "R10",
        "R11",
        "R12",
        "R13",
        "R14",
        "R15",
        "R16",
        "R17",
        "R18",
        "R19",
        "R20",
        "R21",
        "R22",
        "R23",
        "R24",
    },
    "C": {
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
        "C28",
        "C29",
        "C30",
        "C31",
        "C32",
        "C33",
        "C34",
        "C35",
        "C36",
        "C37",
    },
    "E": {
        "E01",
        "E02",
        "E03",
        "E04",
        "E05",
        "E06",
        "E07",
        "E08",
        "E09",
        "E10",
        "E11",
        "E12",
        "E13",
        "E14",
        "E15",
        "E16",
        "E17",
        "E18",
        "E19",
        "E20",
        "E21",
        "E22",
        "E23",
        "E24",
        "E25",
        "E26",
        "E27",
        "E28",
        "E29",
        "E30",
        "E31",
        "E32",
        "E33",
        "E34",
        "E35",
        "E36",
    },
}

# IGS constellation identifiers
constellation_numbering = {1: "G", 3: "R", 5: "C", 6: "E"}

# navigation signal states indicating no measurement ambiguity
required_states = {"G": [1, 8], "R": [1, 8, 128], "C": [1, 8], "E": [1, 8, 2048]}

epsg_gnss_logger = "EPSG:4979"

# tolerance when joining observations to receiver locations
join_tolerance_ms = 990


""" GLONASS FCN to OSN conversion """
fcn_to_osn = {
    "R93": ["R10", "R14"],
    "R94": [],
    "R95": [],
    "R96": ["R02", "R06"],
    "R97": ["R18", "R22"],
    "R98": ["R09", "R13"],
    "R99": ["R12", "R16"],
    "R100": ["R11", "R15"],
    "R101": ["R01", "R05"],
    "R102": ["R20", "R24"],
    "R103": ["R19", "R23"],
    "R104": ["R17", "R21"],
    "R105": ["R03", "R07"],
    "R106": ["R04", "R08"],
}


""" GNSS constants """

# number of nanoseconds in a gnss period
nanos_in_period = {
    "G": 604800 * 10**9,
    "R": 86400 * 10**9,
    "C": 604800 * 10**9,
    "E": 10**8,
}

# offset of gnss epoch from gps
# constellation_epoch_offset = {
#     'G': 0,
#     'R': 86400 * 10**9,
#     'C': 14 * 10**9,
#     'E': -10800*10**9 + 18*10**9,
# }
constellation_epoch_offset = {"G": 0, "R": -3 * 3600 * 10**9, "C": 14 * 10**9, "E": 0}


# lightspeed in m/s
lightspeed = 299792458

""" GPS and UTC conversion"""
nanos_in_day = 86400 * 10**9
nanos_in_year = 315576 * 10**11
nanos_in_minute = 60 * 10**9
nanos_in_second = 10**9

# starting epoch for gps
gps_epoch = pd.to_datetime("1980-01-06", format="%Y-%m-%d")


def leap_seconds(time: pd.Series) -> pd.Series:
    """gps leap seconds."""
    # add to lists as gps seconds announced
    ls_dates_str = ["2015-07-01", "2017-01-01"]
    ls_dates = pd.to_datetime(ls_dates_str, format="%Y-%m-%d")
    ls = np.array([np.nan, 17, 18])
    idx = np.searchsorted(ls_dates, time, side="right")
    if np.min(idx) == 0:
        warnings.warn(f"GPS leap seconds only defined post {ls_dates[0]}")
    output = np.where(pd.isnull(time), np.nan, ls[idx])

    return pd.Series(output, index=time.index, name=time.name).convert_dtypes()


epsg_satellites = "EPSG:4978"
epsg_wgs84 = "EPSG:4979"
epsg_wgs84_cart = "EPSG:4978"
ray_length = 1000
minimum_elevation = 0
maximum_elevation = 85
