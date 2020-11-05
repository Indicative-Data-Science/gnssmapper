"""This script contains functions which will return the location of satellites given coordinates and a date."""

import pandas as pd
import numpy as np
import os
import io
import math
import gzip
import logging
import urllib.request
from datetime import datetime, timezone, date
from collections import defaultdict
from scipy.interpolate import lagrange
from numpy.polynomial import polynomial as P

SP3_LOCAL = "ephemeris"
IONEX_LOCAL = "ionosphere"


def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx


def dateToGPS(utc: str) -> dict:
    """Return a GPS time given a UTC input."""
    datetimeformat = "%Y-%m-%d"
    DATE_GPS_WEEK0 = datetime.strptime("1980-01-06", datetimeformat)
    tdiff = datetime.strptime(utc, datetimeformat) - DATE_GPS_WEEK0
    gpsweek = tdiff.days // 7
    gpsdays = tdiff.days - 7 * gpsweek

    return {'week': gpsweek, 'day': gpsdays}


def getSP3File(date_: str, kind='rapids'):
    SP3_DATASITE = "http://navigation-office.esa.int/products/gnss-products/"
    gpsdate = dateToGPS(date_)

    if kind == 'rapids':
        filenameUltra = 'esu' + str(gpsdate['week']) + str(gpsdate['day']) + '_00.sp3.Z'
        urlUltra = SP3_DATASITE + str(gpsdate['week']) + '/' + filenameUltra
        localUltra = str(os.getcwd()) + '/' + filenameUltra
        urllib.request.urlretrieve(urlUltra, localUltra)
        filetype = filenameUltra

    elif kind=='MGNSS':
        date_list = date_.split('-')
        day_diff = date(int(date_list[0]), int(date_list[1]), int(date_list[2])+1) - date(int(date_list[0]), 1, 1)
        day_diff = day_diff.days
        if len(str(day_diff)) == 1:
            day_diff = '00' + str(day_diff)
        elif len(str(day_diff)) == 2:
            day_diff = '0' + str(day_diff)

        filnameMGNF = 'ESA0MGNFIN_' + str(date_list[0]) + str(day_diff) + '0000_01D_05M_ORB.SP3.gz'
        urlMGNf = SP3_DATASITE + str(gpsdate['week']) + '/' + filnameMGNF
        localUltra = str(os.getcwd()) + '/' + filnameMGNF
        urllib.request.urlretrieve(urlMGNf, localUltra)
        filetype = filnameMGNF

    os.system('uncompress ' + str(filetype))

    return filetype.replace('.Z', '')


def getOrbits(sp3: str, kind='rapids'):
    # from an sp3 text file with format as given here ftp://igs.org/pub/data/format/sp3c.txt
    # returns a DataFrame of locations for each epoch and svid

    def get_time(row: str):
        year = row[3:7]
        month = row[8:10].strip()
        day = row[11:13]
        hour = row[14:16]
        min = row[17:19].strip()
        # sec = row[20:31]

        time_str = year + '-' + month + '-' + day + ' ' + hour + ':' + min + ':' + '00'
        utc_time = datetime.strptime(time_str , '%Y-%m-%d %H:%M:%S')
        return utc_time

    def getXYS(row):
        row_list = row.split()
        svid = row_list[0][1:]
        x = float(row_list[1])*1000
        y = float(row_list[2])*1000
        z = float(row_list[3])*1000
        clockError = float(row_list[4])*1000

        return [svid, x, y, z, clockError]

    if kind == 'MGNSS':
        with gzip.open(sp3, 'rb') as fd:
            file_content = fd.read()
    elif kind == 'rapids':
        with open(sp3, 'rb') as f:
            file_content = f.read()

    sp3 = file_content.decode('utf-8')
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

    output = pd.DataFrame(results, columns=['Epoch', 'UTC Time', 'svid', 'x', 'y', 'z', 'clockError'])
    return output


def getSVIDLocation(sp3_df: pd.DataFrame):
    GPS_UTC_OFFSET = datetime(1980, 1, 6).replace(tzinfo=timezone.utc).timestamp()
    NANOS_TO_SECOND = 1e-9
    sp3_df['UTC Time'] = sp3_df['UTC Time'].apply(lambda x: x.timestamp() - GPS_UTC_OFFSET)
    sp3_df = sp3_df.sort_values(['UTC Time', 'svid'])
    orbitTimes = sp3_df['UTC Time'].unique()
    svids = sp3_df['svid'].unique()
    l1 = len(orbitTimes)
    orbitsXYZ = {}
    for ids in svids:
        orbitsXYZ[ids] = sp3_df.loc[sp3_df['svid'] == ids][sp3_df.columns.difference(['svid', 'clockError', 'Epoch'])]
        orbitsXYZ[ids] = orbitsXYZ[ids].reset_index(drop=True)

    orbitT = defaultdict(list)

    for times in orbitTimes:
        temp = sp3_df.loc[sp3_df['UTC Time'] == times]
        orbitT['UTC Time'].append(times)
        for val in range(len(temp)):
            orbitT[temp.iloc[val,2]].append(temp.iloc[val, -1])

    orbitT = pd.DataFrame.from_dict(orbitT)

    def poly_lagrange(i, alldata: pd.DataFrame):
        if i < 3 or i > len(alldata):
            logging.warning('"outside fit interval"')
        data = alldata.iloc[i-3:i+5, :]
        mid = data.iloc[3,:]
        scale = data.iloc[7, :] - data.iloc[0, :]
        scaled_data = (data - mid)/scale

        polys_svid = {}
        vel_svid = {}
        time_col = scaled_data.iloc[:,0]
        for dfcol in scaled_data.columns:
            if dfcol != 'UTC Time':
                coefs = (P.Polynomial(lagrange(np.array(time_col),
                                                        np.array(scaled_data[dfcol]))).coef)[::-1]
                polys_svid[dfcol] = coefs
                vel_svid[dfcol] = P.polyder(coefs)

        return {i: {'mid': mid, 'scale': scale, 'polys_svid': polys_svid, 'vel_svid': vel_svid}}

    i1 = np.arange(3, l1-3, 4)
    polyXYZ = defaultdict(list)
    for key in orbitsXYZ:
        for i2 in i1:
            polyXYZ[key].append(poly_lagrange(i2, orbitsXYZ[key]))

    def locateSatellite(time, svid):
        if (time is not None) and (svid in svids):
            ind = find_nearest(orbitTimes, time)
            if (ind < 3) or (ind > l1-2):
                logging.warning("outside fit interval")
            j = find_nearest(i1, ind//4)

            poly_dict = polyXYZ[str(svid)][i1[j]][i1[i1[j]]]

            time_to_eval = (time - np.array(poly_dict['mid'])[0])/ np.array(poly_dict['scale'])[0]
            x = P.polyval(time_to_eval, np.array(poly_dict['polys_svid']['x']))
            y = P.polyval(time_to_eval, np.array(poly_dict['polys_svid']['y']))
            z = P.polyval(time_to_eval, np.array(poly_dict['polys_svid']['z']))
            pred_vector = (np.array([x, y, z])*np.array(poly_dict['scale'])[1:]) + np.array(poly_dict['mid'])[1:]

            return pred_vector

        else:
            return {'x': None, 'y': None, 'z': None}

    def velocitySatellite(time, svid):
        if (time is not None) and (svid in svids):
            ind = find_nearest(orbitTimes, time)
            if (ind < 3) or (ind > l1 - 2):
                logging.warning("outside fit interval")
            j = find_nearest(i1, ind // 4)
            poly_dict = polyXYZ[str(svid)][i1[j]][i1[i1[j]]]

            time_to_eval = (time - np.array(poly_dict['mid'])[0]) / np.array(poly_dict['scale'])[0]
            x = P.polyval(time_to_eval, np.array(poly_dict['vel_svid']['x']))
            y = P.polyval(time_to_eval, np.array(poly_dict['vel_svid']['y']))
            z = P.polyval(time_to_eval, np.array(poly_dict['vel_svid']['z']))

            pred_vector = (np.array([x, y, z]) * np.array(poly_dict['scale'])[1:])/np.array(poly_dict['scale'])[0]

            return pred_vector
        else:
            return np.array([None, None, None])

    def rotateECEF(location, time):
        if time is not None and location is not None:
            ROTATION_RATE = 7292115.0 * (10**-11)
            theta = ROTATION_RATE * time
            transformMatrix = np.array([[math.cos(theta), math.sin(theta), 0],
                                        [-math.sin(theta), math.cos(theta), 0],
                                        [0, 0, 1]])
            return np.matmul(transformMatrix, location)
        else:
            return np.array([None, None, None])

    def getRelativisticClockError(location, velocity, svid):
        if svid[1]=='R':
            return 0
        else:
            LIGHTSPEED = 299792458
            dotproduct = np.array(location) @ np.array(velocity)
            return -2 / (LIGHTSPEED**2) * (dotproduct / NANOS_TO_SECOND)

    def getEphemerisClockError(time, svid):
        if (svid in svids) and time is not None:
            sec_time = time*NANOS_TO_SECOND
            timeindex = find_nearest(np.array(orbitT['UTC Time']), sec_time)
            return np.array(orbitT[svid])[timeindex]
        else:
            return None

    def estimateMeasurementTime(receiverGPSTimeNanos, svid):
        # this estimates transmission time for a given epoch signal based on estimated satellite distance of 22,000 km
        # except for Beidou GEO/IGSO which are estimated at 38,000km
        # GPS_UTC_OFFSET <- as.integer(as.POSIXct('1980-01-06',tz="UTC"))
        # GPS_UTC_LEAPSECONDS <- -18
        LIGHTSPEED_NANOS = 0.299792458
        BEIDOU_HIGH_SVID = ["C01", "C02", "C03", "C13", "C16", "C59", "C31", "C04", "C05", "C06", "C07", "C08", "C09",
                           "C10", "C38", "C18", "C39", "C40"]
        BEIDOU_HIGH = 38000000 / LIGHTSPEED_NANOS
        ORBIT = 22000000 / LIGHTSPEED_NANOS
        time_returned = receiverGPSTimeNanos
        if svid in BEIDOU_HIGH_SVID:
            return time_returned - BEIDOU_HIGH
        else:
            return time_returned - ORBIT

    def make_location_calc(transmitterGPSTimeNanos, receiverGPSTimeNanos, svid, only_location=True, isEstimate=False):
        if only_location:
            transmitterGPSTimeNanos = estimateMeasurementTime(receiverGPSTimeNanos, svid)
            locationECEFTransmitted = locateSatellite(transmitterGPSTimeNanos * NANOS_TO_SECOND, svid)
            return locationECEFTransmitted

        if isEstimate:
            transmitterGPSTimeNanos = estimateMeasurementTime(receiverGPSTimeNanos, svid)
            locationECEFTransmitted = locateSatellite(transmitterGPSTimeNanos * NANOS_TO_SECOND, svid)
            result_dict = {
                'time': transmitterGPSTimeNanos,
                'locationECEFTransmitted': locationECEFTransmitted,
                'clockError': None,
                'Rx': None,
                'Ry': None,
                'Rz': None,
                'Vx': None,
                'Vy': None,
                'Vz': None,
                'relativisticClockError': None
            }
            return result_dict
        else:
            clockError = getEphemerisClockError(transmitterGPSTimeNanos, svid)
            transmitterGPSTimeNanos = transmitterGPSTimeNanos - clockError
            locationECEFTransmitted = locateSatellite(transmitterGPSTimeNanos * NANOS_TO_SECOND, svid)
            locationECEFReceived = rotateECEF(locationECEFTransmitted,
                                                (receiverGPSTimeNanos - transmitterGPSTimeNanos) * NANOS_TO_SECOND)
            rotation = locationECEFReceived - locationECEFTransmitted
            velocityECEFTransmitted = velocitySatellite(transmitterGPSTimeNanos * NANOS_TO_SECOND, svid)
            RelativisticClockError = getRelativisticClockError(locationECEFTransmitted, velocityECEFTransmitted, svid)
            transmitterGPSTimeNanos = transmitterGPSTimeNanos - RelativisticClockError
            result_dict = {
                'time': transmitterGPSTimeNanos,
                'locationECEFReceived': locationECEFReceived,
                'clockError': clockError,
                'Rx': rotation[0],
                'Ry': rotation[1],
                'Rz': rotation[2],
                'Vx': velocityECEFTransmitted[0],
                'Vy': velocityECEFTransmitted[1],
                'Vz': velocityECEFTransmitted[2],
                'relativisticClockError': RelativisticClockError
            }
        return result_dict

    return make_location_calc
#
#
# x1 = getSP3File('2020-02-11')
# sp3_file = getOrbits(x1)
# t1 = getSVIDLocation(sp3_file)
# print(t1(None, 1265446185999559200, 'G15'))


