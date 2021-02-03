"""
Module containing constants and definitions used in gnss processing.

Constellation specific information stored as dictionaries.
keys [G,R,C,E] refer to gps,glonasss, beidou, and galileo constellations """

from pandas import to_datetime, DataFrame

""" GNSS logger"""

# minimum version of gnsslogger
minimum_version = "1.4.0.0"

#supported constellations
supported_constellations = set("G","R","C","E")

# IGS constellation identifiers
constellation_numbering = {1: "G", 3: "R", 5: "C", 6: "E"}

# navigation signal states indicating no measurement ambiguity
required_states = {
    'G': [1, 8],
    'R': [1, 8, 128],
    'C': [1, 8],
    'E': [1, 8, 2048],
}

epsg_gnss_logger='EPSG:4979'

""" GNSS constants """

# number of nanoseconds in a gnss period
nanos_in_period = {
    'G': 604800 * 1e9,
    'R': 86400 * 1e9,
    'C': 604800 * 1e9,
    'E': 1e8,
}

# offset of gnss epoch from gps
constellation_epoch_offset = {
    'G': 0,
    'R': 86400 * 1e9,
    'C': 14 * 1e9,
    'E': -10800*1e9 + 18*1e9,
}

# lightspeed in m/s
lightspeed = 299792458

nanos_in_day = 86400 * 1e9

""" GPS and UTC conversion"""
# starting epoch for gps
gps_epoch = pd.to_datetime('1980-01-06', format="%Y-%m-%d")

def leap_seconds(time) -> int:
    """gps leap seconds."""
    # add to lists as gps seconds announced
    ls_dates = ['2015-07-01', '2017-01-01']
    ls = [17, 18]

    ls_ends = ls_dates[1:]+'2100-01-01'
    ls_daterange = [range(to_datetime(s, format="%Y-%m-%d"), to_datetime(f,
                                                                         format="%Y-%m-%d")) for s, f in zip(ls_dates, ls_ends)]
    ls_dic = dict(zip(ls_daterange, ls))
    return ls_dic[time]

epsg_satellites = 'EPSG:4978'
epsg_wgs84 = 'EPSG:4979'
epsg_wgs84_cart = 'EPSG:4978'
ray_length = 1000
minimum_elevation = 0
maximum_elevation = 85