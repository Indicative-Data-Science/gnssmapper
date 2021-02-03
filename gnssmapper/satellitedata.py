""" 
This module contains methods for calculating satellite positions using IGS ephemeris data. It defines a class to avoid reloading the data each time the function is called.

"""

import json
import bisect
import constants
import gpstime
import warnings
from collections import defaultdict, OrderedDict
from typing import Tuple

import pandas as pd
import numpy as np
import os
import re
import io
import math
import gzip
import zlib
import urllib.request
from datetime import datetime, timezone, date
from scipy.interpolate import lagrange
from numpy.polynomial import polynomial as P


class SatelliteData:
    def __init__(self, orbits_filepath: str = "data/orbits/"):
        self.orbits_filepath = orbits_filepath
        self.orbits = defaultdict(dict)

    @property
    def metadata(self) -> set:
        filenames = [f for f in (os.listdir(
            self.orbits_filepath)) if re.match("orbits_", f)]
        days = [re.split('_|\.', f)[1] for f in filenames]
        return set(days)

    def load_orbit(self, day: str) -> dict:
        try:
            with open(self.orbits_filepath+get_filename(day)) as json_file:
                data = json.load(json_file)
            return data

        except IOError:
            return dict()

    def save_orbit(self, day: str, data: dict) -> None:
        with open(self.orbits_filepath+get_filename(day), 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def name_satellites(self, times: pd.Series) -> pd.Series:
        """Provides the svids for satellites tracked in IGS ephemeris data.

        Parameters
        ----------
        time : pd.Series(dtype=int)
            gps time

        Returns
        -------
        pd.Series(dtype=list[str])
            a list of svids visible at each point in time, indexed by time
        """
        # assuming list of svids is static over a day
        days, _ = gpstime.gps_to_doy(time)['date']

        self.update_orbits(days)

        return pd.Series(days.map(lambda x: [self.orbits[x].keys()]), name='svid', index=times)

    def locate_satellites(self, svid: pd.Series, time: pd.Series) -> pd.DataFrame:
        """Returns satellite location in geocentric WGS84 coordinate.

        Calls _locate_sat function, updating ephemeris files if needed.

        Parameters
        ----------
        svid : pd.Series(dtype=str)
            svid in IGS format e.g. "G01"
        time : pd.Series(dtype=int)
            gps time
        Returns
        -------
        pd.DataFrame
            xyz in geocentric wgs84 co-ords 
        """

        doy = gpstime.gps_to_doy(time)
        days = doy['date'],
        nanos = doy['time']
        self.update_orbits(days)
        coords = pd.DataFrame([self._locate_sat(d, t, s)
                               for d, t, s in zip(days, nanos, svid)],columns=['sv_x','sv_y','sv_z'])
        return coords.assign({svid.name:svid.values, time.name:time.values})

    def _locate_sat(self, day: str, time: float, svid: str) -> list:
        """Returns satellite location in geocentric WGS84 coordinates.

        Uses interpolation from a pre-calculated dictionary of coefficients. 

        Parameters
        ----------
        day : str
            day-of-year format
        time : float
            time of day in ns
        svid : str
            satellite ids

        Returns
        -------
        list
            xyz in wgs84 cartesian co-ords
        """

        if day not in self.orbits or svid not in self.orbits[day]:
            warnings.warn(
                f"orbit information not available for {svid} on {day}")
            return [np.nan, np.nan, np.nan]

        # Each day and svid has a nested dictionary of coeffeicients covering different periods of the day.
        # Select the one that has a key closest to the required time.
        # ordered dictionary in sorted order for keys
        keys = [float(x) for x in self.orbits[day][svid]]
        close = bisect.bisect_left(keys, time)
        if close == 0:
            idx = close
        elif close == len(keys):
            idx = close-1
        else:
            idx = min([close-1, close], key=lambda x: abs(keys[x]-time))

        poly_dict = self.orbits[day][svid][list(self.orbits[day][svid])[idx]]

        if time > poly_dict['ub'] or time < poly_dict['lb']:
            warnings.warn(
                f"Orbits available for {svid} on {day}, however a valid dictionary wasn't found at {time}")
            return [np.nan, np.nan, np.nan]

        scaled_time = (time - poly_dict['mid'][0]) / poly_dict['scale'][0]

        def predict(dim):
            n = ['x', 'y', 'z'].index(dim) + 1
            return P.polyval(scaled_time, poly_dict[dim]) * poly_dict['scale'][n] + poly_dict['mid'][n]

        return [predict(dim) for dim in ['x', 'y', 'z']]

    def update_orbits(self, days: pd.Series) -> None:
        """Loads orbits, updating orbit database where necessary.

        Parameters
        ----------
        days : pd.Series
            days in YYYYdoy string format 
        """
        days = set(days.unique())
        missing_days = days - self.metadata

        for day in missing_days:
            orbit_dic = defaultdict(dict)
            sp3 = get_SP3_file(day)
            df = get_SP3_dataframe(sp3)
            unsorted_orbit = create_orbit(df)
            for svid, dic in unsorted_orbit.items():
                orbit_dic[svid] = OrderedDict(sorted(dic.items()))
            self.save_orbit(day, orbit_dic)

        for day in days:
            if day not in self.orbits:
                self.orbits[day] = self.load_orbit(day)


def create_orbit(sp3_df: pd.DataFrame) -> dict:
    """Creates a dictionary of Lagrangian 7th order polynomial coefficents used for interpolation. 

    Estimates centred at every 4th data point, including lower and upper time bounds of validity, scaling parameters and lagrangian coeffecients.

    Parameters
    ----------
    sp3_df : pd.DataFrame
        includes x,y,z and time columns

    Returns
    -------
    dict
        {id:{midpoint of period: {dictionary of estimates}}}
    """
    day = sp3_df['date'].str[:4].astype(int)
    sp3_df['gpstime'] = sp3_df['time'] + (day-min(day)) * 24 * 3600 * 1e9
    sp3_df = sp3_df.sort_values(['svid', 'gpstime'])
    polyXYZ = defaultdict(dict)

    for id_ in sp3_df['svid'].unique():
        orbits = sp3_df.loc[sp3_df['svid'] == id_, ['gpstime', 'x', 'y', 'z']]
        idxs = list(range(3, len(orbits)-5, 4))
        if idxs[-1] != len(orbits)-5:
            idxs.append(len(orbits)-5)
        for i in idxs:
            k, v = poly_lagrange(i, orbits)
            polyXYZ[id_][k] = v

    return polyXYZ


def poly_lagrange(i: int, alldata: pd.DataFrame) -> Tuple[float, dict]:
    """Returns lagrangian polynomial coefficients, along with scaling parameters used to avoid problems with numerically small coeffecients.

    Parameters
    ----------
    i : int
        index of observation
    alldata : pd.DataFrame
        periodic observations of 'gpstime', 'x', 'y', 'z', all simple floats

    Returns
    -------
    Tuple[float,dict]
        float: midpoint of time validity
        dict: lower and upper time bounds of validity, scaling parameters and lagrangian coeffecients
    """
    if i < 3 or i > len(alldata)-5:
        raise ValueError('"outside fit interval"')
    data = alldata.iloc[i-3:i+5, :].reset_index(drop=True)
    lb, ub = data.iloc[0, 0], data.iloc[7, 0]
    mid = data.iloc[3, :]
    scale = data.iloc[7, :] - data.iloc[0, :]
    scaled_data = (data - mid)/scale

    def coefs(dim: str) -> list:
        # why turned round
        return np.asarray(lagrange(scaled_data["gpstime"], scaled_data[dim])).tolist()[::-1]

    return [mid[0], {'lb': lb, 'ub': ub, 'mid': list(mid), 'scale': list(scale), 'x': coefs('x'), 'y': coefs('y'), 'z': coefs('z')}]


def get_filename(day: str) -> str:

    return "orbits_"+day+".json"


def get_SP3_file(date: str, orbit_type='MGNSS') -> str:
    """Loads a file of precise satellite orbits. 

    Checks for local saved version otherwise fetches remotely. 

    Parameters
    ----------
    date : str
        date in YYYYdoy format
    orbit_type : str, optional
        type of precise orbit product {MGNSS, rapids}, by default 'MGNSS'

    Returns
    -------
    str
        orbit file contents in string format

    """
    SP3_DATASITE = "http://navigation-office.esa.int/products/gnss-products/"
    time = gpstime.doy_to_gps(pd.Series([date]), pd.Series([0]))
    gpsdate = gpstime.gps_to_gpsweek(time)[0]

    if orbit_type not in ['rapids', 'MGNSS']:
        raise ValueError('invalid orbit type selected')

    if orbit_type == 'rapids':
        filename = 'esu' + str(gpsdate['week']) + \
            str(gpsdate['day']) + '_00.sp3.Z'
    elif orbit_type == 'MGNSS':
        filename = 'ESA0MGNFIN_' + date + '0000_01D_05M_ORB.SP3.gz'

    url = SP3_DATASITE + str(gpsdate['week']) + '/' + filename
    local = str(os.getcwd()) + '/data/sp3/' + filename
    if not os.path.isfile(local):
        urllib.request.urlretrieve(url, local)
    # filetype = filename
    # os.system('uncompress ' + str(filetype))
    # return filetype.replace('.Z', '')

    if orbit_type == 'MGNSS':
        with gzip.open(local, 'rt', encoding='utf-8') as f:
            file_content = f.read()

    elif orbit_type == 'rapids':
        with open(local, 'rb') as f:
            binary_content = zlib.decompress(f.read())
            file_content = binary_content.decode('utf-8')

    return file_content


def get_SP3_dataframe(sp3: str) -> pd.DataFrame:
    """Parse a precise orbits string to retrieve orbit data.

    Format as given here ftp://igs.org/pub/data/format/sp3c.txt

    Parameters
    ----------
    sp3 : str
        precise orbits, e.g. read from a txt file.

    Returns
    -------
    pd.DataFrame
        locations for each epoch and svid
    """

    def get_date(row: str):
        date = pd.Timestamp(year=row[3:7], month=row[8:10], day=row[11:13])
        year = date.year.astype("str")
        day = date.dayofyear.astype("str")
        return year.str.cat(day.str.pad(3, side='left', fillchar="0"))

    def get_time(row: str):
        hour = int(row[14:16])
        min = int(row[17:19])
        sec = int(row[20:31])
        time = hour * 3600 + min * 60 + sec
        return time * 1e9

    def getXYS(row):
        row_list = row.split()
        svid = row_list[0][1:]
        x = float(row_list[1])*1000
        y = float(row_list[2])*1000
        z = float(row_list[3])*1000
        clockError = float(row_list[4])*1000

        return [svid, x, y, z, clockError]

    buf = io.StringIO(sp3)

    results = []
    epoch = 0

    while True:
        nstr = buf.readline()
        if len(nstr) == 0:
            break
        if nstr[:2] == '* ':
            date = get_date(nstr)
            time = get_time(nstr)
            epoch += 1
        if nstr[0] == 'P':
            pos = getXYS(nstr)
            df_row = [epoch, date, time] + pos
            results.append(df_row)

    output = pd.DataFrame(
        results, columns=['epoch', 'date', 'time', 'svid', 'x', 'y', 'z', 'clockerror'])
    return output
