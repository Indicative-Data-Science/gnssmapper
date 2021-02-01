"""
Module containing constants and definitions used in gnss processing.

Constellation specific information stored as dictionaries.
keys [G,R,C,E] refer to gps,glonasss, beidou, and galileo constellations """

from pandas import as_datetime, DataFrame


# number of nanonseconds in a gnss period
nanos_in_period = {
    'G': 604800 * 1e9,
    'R': 86400 * 1e9,
    'C': 604800 * 1e9,
    'E': 1e8,
}

# offset of gnss epoch from gps
epoch_offset_nanos = {
    'G': 0,
    'R': 86400 * 1e9,
    'C': 14 * 1e9,
    'E': -10800*1e9 + 18*1e9,
}

# starting epoch for gps
gps_epoch = as_datetime('1980-01-06', format="%Y-%m-%d")

# lightspeed in m/s
lightspeed = 299792458


def leap_seconds(time) -> int:
    """gps leap seconds."""
    # add to lists as gps seconds announced
    ls_dates = ['2015-07-01', '2017-01-01']
    ls = [17, 18]

    ls_ends = ls_dates[1:]+'2100-01-01'
    ls_daterange = [range(as_datetime(s, format="%Y-%m-%d"), as_datetime(f,
                                                                         format="%Y-%m-%d")) for s, f in zip(ls_dates, ls_ends)]
    ls_dic = dict(zip(ls_daterange, ls))
    return ls_dic[time]
