""" 
=========================================
GNSS Data 
=========================================

This module calculates GNSS signal information from GNSS raw data

"""

import json
import bisect
from collections import OrderedDict

import pandas as pd
import numpy as np
import os
import io
import math
import gzip
import zlib
import urllib.request
from datetime import datetime, timezone, date
from collections import defaultdict
from scipy.interpolate import lagrange
from numpy.polynomial import polynomial as P

class GNSSData:
    def __init__(self,orbits_filepath="data/orbits/orbits.json",meta_filepath="data/orbits/orbits_meta.json"):
        self.orbits_filepath = orbits_filepath
        self.meta_filepath = meta_filepath
        self.orbits = self.load_orbits(orbits_filepath)
        self.metadata = self.load_meta(meta_filepath)

    @staticmethod
    def load_orbits(file:str ) -> dict:
        """ retrieves the file of orbits data 
        """
        try:
            with open(file) as json_file:
                data = json.load(json_file)
            return data

        except IOError:
            return defaultdict(dict)

    @staticmethod    
    def load_meta(file:str) -> dict:
        """ retrieves the file of orbits metadata 
        """
        try:
            with open(file) as json_file:
                data = json.load(json_file)
            return data

        except IOError:
            return set()

    def save(self) ->None:
        """save data
        """
        with open(self.orbits_filepath, 'w') as outfile:
            json.dump(self.orbits, outfile,indent=4)
        with open(self.meta_filepath, 'w') as outfile:
            json.dump(self.metadata, outfile,indent=4)

    def satLocation(self,sv,times):
        """ returns satellite locations
        Parameters
        ----------
        sv : [n,] string array 
            satellite ids
        times: [n,] np.datetime64 array
            time of observation
        Returns
        -------
        location : [n,3] array 
            location in wgs84 cartesian co-ords 
        """
        transmitted = estimateMeasurementTime(times,sv) #vectorise  

        days = set(np.datetime_as_string(transmitted,unit='D').flatten())
        missing_days = days - self.metadata 
        self.updateOrbits(missing_days)
        self.save()

        gpstimes = utc_to_gps(transmitted)
        return np.array([self.locateSatellite(t,s) for t,s in zip(gpstimes,sv)])

    def locateSatellite(self,time: float, svid: str)-> list:
        """ returns satellite location
        Parameters
        ----------
        svid: satellite ids
        time: seconds since gps epoch 
        Returns
        -------
        location : xyz in wgs84 cartesian co-ords 
        """
        
        if svid not in self.orbits:
            return [np.nan,np.nan.np.nan]

        idx = bisect.bisect_left(list(self.orbits[svid]),time) #requires ordered dictionary in sorted order for keys
        poly_dict = self.orbits[svid][idx]

        if time > poly_dict['ub'] or time < poly_dict['lb']:
            return [np.nan,np.nan.np.nan]
    
        scaled_time = (time - poly_dict['mid'][0]) / poly_dict['scale'][0]

        def predict(dim):
            n = ['x','y','z'].index(dim) + 1
            return P.polyval(scaled_time, poly_dict[dim]) *poly_dict['scale'][n] + poly_dict['mid'][n]

        return [predict(dim) for dim in ['x','y','z']]

    def updateOrbits(self,days):
        """ updates orbit database for missing days
        Parameters
        ----------
        days : set of string utc dates.
        """
        unsorted_orbits = self.orbits

        for day in days:
            sp3 = getSP3File(day)
            orbits = getOrbits(sp3)
            lagrangian_dic = createLagrangian(orbits)
            for sv,dic in lagrangian_dic.items():
                for time,coefs_dic in dic.items():
                    unsorted_orbits[sv][time]= coefs_dic
        

        sorted_orbits={}
        for sv,dic in unsorted_orbits.items():
            sorted_orbits[sv]= OrderedDict(sorted(dic.items()))
        
        self.orbits=sorted_orbits
        self.metadata += days
        

def createLagrangian(sp3_df: pd.DataFrame):
    # creates Lagrangian 7th order polynomials for estimating location, generating estimates centred at every 4th data point
    sp3_df['gpstime'] = utc_to_gps(sp3_df['utctime'])
    sp3_df = sp3_df.sort_values(['gpstime', 'svid'])
    polyXYZ = defaultdict(dict)

    for id_ in sp3_df['svid'].unique():
        orbits = sp3_df.loc[sp3_df['svid'] == id_,['gpstime','x','y','z']]
        for i in range(3, len(orbits)-5, 4):
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


def estimateMeasurementTime(time, svid):
        # this estimates transmission time for a given epoch signal based on estimated satellite distance of 22,000 km
        # except for Beidou GEO/IGSO which are estimated at 38,000km
        # GPS_UTC_OFFSET <- as.integer(as.POSIXct('1980-01-06',tz="UTC"))
        # GPS_UTC_LEAPSECONDS <- -18
        LIGHTSPEED = 299792458
        BEIDOU_HIGH_SVID = ["C01", "C02", "C03", "C13", "C16", "C59", "C31", "C04", "C05", "C06", "C07", "C08", "C09",
                           "C10", "C38", "C18", "C39", "C40"]
        BEIDOU_HIGH = 38000000 / LIGHTSPEED
        ORBIT = 22000000 / LIGHTSPEED
        return np.where(np.isin(svid,BEIDOU_HIGH_SVID), time - BEIDOU_HIGH, time - ORBIT)
     

def utc_to_gps(time):
    """Returns a GPS timestamp since epoch give a np.datetime64 input"""
    gps_epoch = np.datetime64("1980-01-06")
    delta= time- gps_epoch
    return np.array([np.timedelta64(t,'ns') for t in delta],dtype=float) 

def dateToGPS(utc: str) -> dict:
    """Return a GPS time given a UTC input."""
    gps_epoch = np.datetime64("1980-01-06") 
    tdiff= np.array(np.timedelta64(utc-gps_epoch,'D'),dtype=float)

    # datetimeformat = "%Y-%m-%d"
    # DATE_GPS_WEEK0 = datetime.strptime("1980-01-06", datetimeformat)
    # tdiff = datetime.strptime(utc, datetimeformat) - DATE_GPS_WEEK0
    gpsweek = tdiff // 7
    gpsdays = tdiff - 7 * gpsweek

    return {'week': gpsweek, 'day': gpsdays}

def getSP3File(date_: str, kind='MGNSS'):
    """ obtains a file of satellite orbits, fetching remotely if needed. 
    
    Returns
    -------
    text file with given format ftp://igs.org/pub/data/format/sp3c.txt

    """
    SP3_DATASITE = "http://navigation-office.esa.int/products/gnss-products/"
    gpsdate = dateToGPS(np.datetime64(date_,"D"))

    if kind == 'rapids':
        filename= 'esu' + str(gpsdate['week']) + str(gpsdate['day']) + '_00.sp3.Z'
    elif kind=='MGNSS':
        date_list = date_.split('-')
        day_diff = date(int(date_list[0]), int(date_list[1]), int(date_list[2])+1) - date(int(date_list[0]), 1, 1)
        day_diff = day_diff.days
        if len(str(day_diff)) == 1:
            day_diff = '00' + str(day_diff)
        elif len(str(day_diff)) == 2:
            day_diff = '0' + str(day_diff)

        filename = 'ESA0MGNFIN_' + str(date_list[0]) + str(day_diff) + '0000_01D_05M_ORB.SP3.gz'
       
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

def getOrbits(sp3: str, kind='MGNSS'):
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

