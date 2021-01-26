""" 
=========================================
GNSS Data 
=========================================

This module calculates GNSS signal information from GNSS raw data. It defines a class to avoid reloading the data each time the function is called.

"""

import json
import bisect
from collections import defaultdict,OrderedDict

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


class GNSSData:
    def __init__(self,orbits_filepath="data/orbits/"):
        self.orbits_filepath = orbits_filepath
        self.orbits= defaultdict(dict)

    @property
    def metadata(self):
        filenames = [ f for f in (os.listdir(self.orbits_filepath)) if re.match("orbits_",f)]
        days=[re.split('_|\.',f)[1] for f in filenames]
        
        return set(days)

    def load_orbit(self,day):
        try:
            with open(self.orbits_filepath+get_filename(day)) as json_file:
                data = json.load(json_file)
            return data

        except IOError:
            return dict()

    def save_orbit(self,day,data) ->None:
        with open(self.orbits_filepath+get_filename(day), 'w') as outfile:
            json.dump(data, outfile,indent=4)


    def locate_satellites(self,sv,times):
        """ returns satellite locations
        Parameters
        ----------
        sv : [n,] string array 
            satellite ids
        times: [n,] np.datetime64 array
            utc time of observation
        Returns
        -------
        location : [n,3] array 
            location in wgs84 cartesian co-ords 
        """
        transmitted = estimate_measurement_time(times,sv) #vectorise  
        days,gpstimes = utc_to_dayofyear(transmitted)
        self.update_orbits(days)
        return np.array([self.locate_sat(d,t,s) for d,t,s in zip(days,gpstimes,sv)])

    def locate_sat(self,day:str, time: float, svid: str)-> list:
        """ returns satellite location
        Parameters
        ----------
        svid: satellite ids
        day: doy format 
        time: time of day in ns 
        Returns
        -------
        location : xyz in wgs84 cartesian co-ords 
        """
        
        if day not in self.orbits or svid not in self.orbits[day]:
            return [np.nan,np.nan,np.nan]

        keys= [float(x) for x in self.orbits[day][svid]]
        close = bisect.bisect_left(keys,time) #requires ordered dictionary in sorted order for keys
        if close==0:
            idx = close 
        elif close ==len(keys):
            idx = close-1
        else:
            idx = min([close-1,close],key= lambda x: abs(keys[x]-time))
            
        poly_dict = self.orbits[day][svid][list(self.orbits[day][svid])[idx]]

        if time > poly_dict['ub'] or time < poly_dict['lb']:
            return [np.nan,np.nan,np.nan]
    
        scaled_time = (time - poly_dict['mid'][0]) / poly_dict['scale'][0]

        def predict(dim):
            n = ['x','y','z'].index(dim) + 1
            return P.polyval(scaled_time, poly_dict[dim]) *poly_dict['scale'][n] + poly_dict['mid'][n]

        return [predict(dim) for dim in ['x','y','z']]

    def update_orbits(self,days):
        """ loads orbits, updating orbit database where necessary
        Parameters
        ----------
        days : np.array of string days in doy format.
        """
        days = set(days.flatten())
        missing_days = days - self.metadata

        for day in missing_days:
            orbit_dic=defaultdict(dict)
            sp3 = get_SP3_file(day)
            df = get_SP3_dataframe(sp3)
            unsorted_orbit = create_orbit(df)
            for sv,dic in unsorted_orbit.items():
                orbit_dic[sv]=OrderedDict(sorted(dic.items()))
            self.save_orbit(day,orbit_dic)
  
        for day in days:
            if day not in self.orbits:  
                self.orbits[day]=self.load_orbit(day)
        
def create_orbit(sp3_df: pd.DataFrame):
    # creates Lagrangian 7th order polynomials for estimating location, generating estimates centred at every 4th data point
    daystr,sp3_df['gpstime'] = utc_to_dayofyear(sp3_df['utctime'])
    day = daystr.astype(int)
    sp3_df['gpstime'] = sp3_df['gpstime']  + (day-min(day)) * 24* 3600 *1e9
    sp3_df = sp3_df.sort_values(['svid','gpstime'])
    polyXYZ = defaultdict(dict)

    for id_ in sp3_df['svid'].unique():
        orbits = sp3_df.loc[sp3_df['svid'] == id_,['gpstime','x','y','z']]
        idxs = list(range(3, len(orbits)-5, 4))
        if idxs[-1]!=len(orbits)-5:
            idxs.append(len(orbits)-5)
        for i in idxs:
            k,v = poly_lagrange(i, orbits )
            polyXYZ[id_][k] = v
    
    return polyXYZ

def poly_lagrange(i, alldata: pd.DataFrame):
    """ returns lagrangian polynomial coefficients, along with scaling parameters used to avoid problems with numerically small coeffecients
    Parameters
    ----------
    i : int 
        index of observation
    alldata: [n,] pd.DataFrame with columns of  'gpstime', 'x', 'y', 'z', all simple floats
        time of observation
    Returns
    -------
    lower and upper time bounds of validity, scaling parameters and lagrangian coeffecients
    
    """
    if i < 3 or i > len(alldata)-5:
        raise ValueError('"outside fit interval"')
    data = alldata.iloc[i-3:i+5, :].reset_index(drop=True)
    lb, ub = data.iloc[0,0], data.iloc[7,0] 
    mid = data.iloc[3,:]
    scale = data.iloc[7, :] - data.iloc[0, :]
    scaled_data = (data - mid)/scale

    def coefs(dim: str) -> list:
        return np.asarray(lagrange(scaled_data["gpstime"],scaled_data[dim])).tolist()[::-1] # why turned round

    return [mid[0], {'lb': lb, 'ub': ub, 'mid': list(mid), 'scale': list(scale), 'x': coefs('x'), 'y': coefs('y'), 'z': coefs('z') } ]

def estimate_measurement_time(time, svid):
        # this estimates transmission time for a given epoch signal based on estimated satellite distance of 22,000 km
        # except for Beidou GEO/IGSO which are estimated at 38,000km
        # GPS_UTC_OFFSET <- as.integer(as.POSIXct('1980-01-06',tz="UTC"))
        # GPS_UTC_LEAPSECONDS <- -18
        LIGHTSPEED = 0.299792458 #metres per ns
        BEIDOU_HIGH_SVID = ["C01", "C02", "C03", "C13", "C16", "C59", "C31", "C04", "C05", "C06", "C07", "C08", "C09",
                           "C10", "C38", "C18", "C39", "C40"]
        BEIDOU_HIGH = np.timedelta64(int(38000000 / LIGHTSPEED),'ns')
        ORBIT = np.timedelta64(int(22000000 / LIGHTSPEED),'ns')
        return np.where(np.isin(svid,BEIDOU_HIGH_SVID), time - BEIDOU_HIGH, time - ORBIT)
     
def get_filename(day:str)->str:
    return "orbits_"+day+".json"

def utc_to_dayofyear(dates):
    """Returns a doy format"""
    year = np.datetime_as_string(dates,"Y")
    doy = np.array([np.datetime64(date,"D") - np.datetime64(date,"Y")+1 for date in dates]).astype(int).astype(str)
    ns = np.array([np.datetime64(date,"ns") - np.datetime64(date,"D") for date in dates]).astype(float)
    yeardoy= np.char.add(year,np.char.zfill(doy,3))
    return [yeardoy,ns]

def dayofyear_to_GPS(yeardoy: str) -> dict:
    """Return a GPS date given a doy input."""
    year,doy = np.datetime64(yeardoy[:4],"Y"),yeardoy[4:]
    gps_epoch = np.datetime64("1980-01-06","D") 
    tdiff= (year-gps_epoch+np.timedelta64(int(doy)-1,'D')).astype(int)
    gpsweek = tdiff // 7
    gpsdays = tdiff - 7 * gpsweek

    return {'week': gpsweek, 'day': gpsdays}

def get_SP3_file(date_: str, kind='MGNSS'):
    """ obtains a file of satellite orbits, fetching remotely if needed. 
    
    Returns
    -------
    text file with given format ftp://igs.org/pub/data/format/sp3c.txt

    """
    SP3_DATASITE = "http://navigation-office.esa.int/products/gnss-products/"
    gpsdate = dayofyear_to_GPS(date_)

    if kind == 'rapids':
        filename= 'esu' + str(gpsdate['week']) + str(gpsdate['day']) + '_00.sp3.Z'
    elif kind=='MGNSS':
        filename = 'ESA0MGNFIN_' + date_ + '0000_01D_05M_ORB.SP3.gz'
       
    url = SP3_DATASITE + str(gpsdate['week']) + '/' + filename
    local = str(os.getcwd()) + '/data/sp3/' + filename
    if not os.path.isfile(local):
        urllib.request.urlretrieve(url, local)
    # filetype = filename
    # os.system('uncompress ' + str(filetype))
    # return filetype.replace('.Z', '')

    if kind == 'MGNSS':
        with gzip.open(local, 'rt',encoding='utf-8') as f:
            file_content = f.read()

    elif kind == 'rapids':
        with open(local, 'rb') as f:
            binary_content = zlib.decompress(f.read())
            file_content = binary_content.decode('utf-8')

    return file_content

def get_SP3_dataframe(sp3: str, kind='MGNSS'):
    """ parse a precise orbits txt file to retrieve orbit data
    Parameters
    ----------
    format as given here ftp://igs.org/pub/data/format/sp3c.txt

    Returns
    -------
    DataFrame [n,7] with columns of locations for each epoch and svid

    """

    def get_time(row: str):
        year = row[3:7]
        month = row[8:10].strip().zfill(2)
        day = row[11:13].strip().zfill(2)
        hour = row[14:16].strip().zfill(2)
        min = row[17:19].strip().zfill(2)
        sec = row[20:31].strip().zfill(11)
        time_str = year + '-' + month + '-' + day + 'T' + hour + ':' + min + ':' + sec
        utc_time = np.datetime64(time_str , 'ns')
        return utc_time

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
            utctime = get_time(nstr)
            epoch += 1
        if nstr[0] == 'P':
            pos = getXYS(nstr)
            df_row = [epoch, utctime] + pos
            results.append(df_row)

    output = pd.DataFrame(results, columns=['epoch', 'utctime', 'svid', 'x', 'y', 'z', 'clockerror'])
    return output

