""" 
This module contains methods for calculating satellite positions using IGS ephemeris data_var. It defines a class to avoid reloading the data_var each time the function is called.

"""

import bisect
from collections import defaultdict, OrderedDict
import gzip
import importlib.resources
import json
import os
import re
import urllib.request
import warnings
import zlib


import numpy as np
from numpy.polynomial import polynomial as P
import pandas as pd
from scipy.interpolate import lagrange

import gnssmapper.common as cm
import gnssmapper.data as data


class SatelliteData:
    def __init__(self):
        self.orbits = defaultdict(dict)

    @property
    def metadata(self) -> set:
        filenames = [f for f in importlib.resources.contents(data.orbits) if re.match("orbits_", f)]
        days = [re.split(r'_|\.', f)[1] for f in filenames]
        return set(days)

    def _load_orbit(self, day: str) -> dict:
        try:
            with importlib.resources.open_binary(data.orbits,_get_filename(day)) as json_file:
                data_var = json.load(json_file)
            return data_var

        except IOError:
            return dict()

    def _save_orbit(self, day: str, data_var: dict) -> None:
        with open(data.orbits.__path__[0]+'/'+_get_filename(day), 'w') as outfile:
            json.dump(data_var, outfile, indent=4)

    def name_satellites(self, time: pd.Series) -> pd.Series:
        """Provides the svids for satellites tracked in IGS ephemeris data_var.

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
        days = cm.time.gps_to_doy(time)['date']

        self._update_orbits(days)
        svids = pd.Series(days.map(lambda x: list(self.orbits[x].keys())), name='svid')
        svids.index=time
        return svids

    def locate_satellites(self, svid: pd.Series, time: pd.Series) -> pd.DataFrame:
        """Returns satellite location in geocentric WGS84 coordinates.

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

        doy = cm.time.gps_to_doy(time)
        days = doy['date']
        nanos = doy['time']
        self._update_orbits(days)
        coords = pd.DataFrame([self._locate_sat(d, t, s)
                               for d, t, s in zip(days, nanos, svid)], columns=['sv_x', 'sv_y', 'sv_z'])
        new_columns = {svid.name:svid.array,time.name:time.array}
        return coords.assign(**new_columns)

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

    def _update_orbits(self, days: pd.Series) -> None:
        """Loads orbits, updating orbit database where necessary.

        Parameters
        ----------
        days : pd.Series
            days in YYYYdoy string format 
        """
        days_ = set(days)
        missing_days = days_ - self.metadata

        if missing_days:
            print(f"{missing_days} orbits are missing and must be created.")
        for day in missing_days:
            orbit_dic = defaultdict(dict)
            print(f"downloading sp3 file for {day}.")
            sp3 = _get_sp3_file(day)
            df = _get_sp3_dataframe(sp3)
            print(f"creating {day} orbit.")
            unsorted_orbit = _create_orbit(df)
            for svid, dic in unsorted_orbit.items():
                orbit_dic[svid] = OrderedDict(sorted(dic.items()))
            print(f"saving {day} orbit.")
            self._save_orbit(day, orbit_dic)

        for day in days_:
            if day not in self.orbits:
                self.orbits[day] = self._load_orbit(day)


def _create_orbit(sp3_df: pd.DataFrame) -> dict:
    """Creates a dictionary of Lagrangian 7th order polynomial coefficents used for interpolation. 

    Estimates centred at every 4th data_var point, including lower and upper time bounds of validity, scaling parameters and lagrangian coeffecients.

    Parameters
    ----------
    sp3_df : pd.DataFrame
        includes x,y,z and time columns

    Returns
    -------
    dict
        {id:{midpoint of period: {dictionary of estimates}}}
    """
    day = sp3_df['date'].str[4:].astype(int)
    sp3_df['tm'] = sp3_df['time'].astype(int) + (day-min(day)) * 24 * 3600 * 10**9
    sp3_df = sp3_df.sort_values(['svid', 'tm'])
    polyXYZ = defaultdict(dict)

    for id_ in sp3_df['svid'].unique():
        orbits = sp3_df.loc[sp3_df['svid'] == id_, ['tm', 'x', 'y', 'z']]
        idxs = list(range(3, len(orbits)-5, 4))
        if idxs[-1] != len(orbits)-5:
            idxs.append(len(orbits)-5)
        for i in idxs:
            k, v = _poly_lagrange(i, orbits)
            polyXYZ[id_][k] = v

    return polyXYZ


def _poly_lagrange(i: int, alldata: pd.DataFrame) -> list[float, dict]:
    """Returns lagrangian polynomial coefficients, along with scaling parameters used to avoid problems with numerically small coeffecients.

    Parameters
    ----------
    i : int
        index of observation
    alldata : pd.DataFrame
        periodic observations of 'tm', 'x', 'y', 'z', all simple floats

    Returns
    -------
    list[float,dict]
        float: midpoint of time validity
        dict: lower and upper time bounds of validity, scaling parameters and lagrangian coeffecients
    """
    if i < 3 or i > len(alldata)-5:
        raise ValueError('"outside fit interval"')
    data_var = alldata.iloc[i-3:i+5, :].reset_index(drop=True)
    lb, ub = data_var.iloc[0, 0], data_var.iloc[7, 0]
    mid = data_var.iloc[3, :]
    scale = data_var.iloc[7, :] - data_var.iloc[0, :]
    scaled_data = (data_var - mid)/scale

    def coefs(dim: str) -> list:
        # why turned round
        return np.asarray(lagrange(scaled_data["tm"], scaled_data[dim])).tolist()[::-1]

    return [mid.tolist()[0], {'lb': lb.tolist(), 'ub': ub.tolist(), 'mid': mid.tolist(), 'scale': scale.tolist(), 'x': coefs('x'), 'y': coefs('y'), 'z': coefs('z')}]


def _get_filename(day: str) -> str:

    return "orbits_"+day+".json"


def _get_sp3_file(date: str, orbit_type='final') -> str:
    """Loads a file of precise satellite orbits. 

    Checks for local saved version otherwise fetches remotely. 

    Parameters
    ----------
    date : str
        date in YYYYdoy format
    orbit_type : str, optional
        type of precise orbit product {final, rapid}, by default 'final'

    Returns
    -------
    str
        orbit file contents in string format

    """
    

    filename = _sp3_filename[orbit_type](date)
    filename
    if not importlib.resources.is_resource(data.sp3, filename):
        url = _sp3_datasite + _sp3_filepath(date) + filename
        local_path = data.sp3.__path__[0]+'/'+filename
        urllib.request.urlretrieve(url, local_path)
        
    zipfile = importlib.resources.read_binary(data.sp3, filename)
    
    extension = filename.rsplit(".", maxsplit=1)[1]
    if extension == 'gz':
        binary = gzip.decompress(zipfile)
    else:
        binary = zlib.decompress | (zipfile)
    txt = binary.decode('utf-8')

    return txt


def _get_sp3_dataframe(sp3: str) -> pd.DataFrame:
    """Parse a precise orbits string to retrieve orbit data_var.

    Format as given here ftp://igs.org/pub/data_var/format/sp3c.txt

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
        _utc = pd.Timestamp(year=int(row[3:7]), month=int(row[8:10]), day=int(row[11:13]))
        year = str(_utc.year)
        day = str(_utc.dayofyear)
        return year + day.zfill(3)

    def get_time(row: str):
        hour = int(row[14:16])
        min = int(row[17:19])
        sec = int(float(row[20:31]))
        time = hour * 3600 + min * 60 + sec
        return time * 10**9

    def getXYS(row):
        row_list = row.split()
        svid = row_list[0][1:]
        x = float(row_list[1]) * 1000
        y = float(row_list[2]) * 1000
        z = float(row_list[3]) * 1000
        clockError = float(row_list[4]) * 1000

        return [svid, x, y, z, clockError]

    lines = sp3.splitlines()

    results = []
    epoch = 0

    for line in lines:
        if line[:2] == '* ':
            date = get_date(line)
            time = get_time(line)
            epoch += 1
        if line[0] == 'P':
            pos = getXYS(line)
            df_row = [epoch, date, time] + pos
            results.append(df_row)

    output = pd.DataFrame(
        results, columns=['epoch', 'date', 'time', 'svid', 'x', 'y', 'z', 'clockerror'])
    return output


""" SP3 download related functions """

_sp3_datasite = "http://navigation-office.esa.int/products/gnss-products/"
def _sp3_filepath(date):
    return str(_sp3_filename_date_conversion(date)['week'][0])+'/'

_sp3_filename = {
    'rapid': lambda x: 'esu' + str(_sp3_filename_date_conversion(x)['week'][0]) + str(_sp3_filename_date_conversion(x)['day'][0]) + '_00.sp3.Z',
    'final': lambda x: 'ESA0MGNFIN_' + x + '0000_01D_05M_ORB.SP3.gz'}

def _sp3_filename_date_conversion(date):
    """ Converts a doy format to gps week  """
    time = cm.time.doy_to_gps(pd.Series([date]), pd.Series([0]))
    return cm.time.gps_to_gpsweek(time)
