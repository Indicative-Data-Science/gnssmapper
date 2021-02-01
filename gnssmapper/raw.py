"""
Module contains methods for processing raw GNSS data eg from gnss logger.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from math import floor
import constants
import warnings
from typing import Tuple


# readLog <- function(file,calculatepr,...){
#     cat("Reading",file,"\n")
#     data <- readGnss(file) %>% preprocessGnss(calculatepr)
#     epochs <- getEpochs(data,...)
#     NumberAllEpochs <- NROW(epochs)
#     numberAllSignals <- NROW(data[["Raw"]])
#     epochs <- epochs[!missing(epochs) & !is.na(epochs$UTCTime),] #ensures valid time and position measurement
#     measurements <- data[["Raw"]] %>% filter(EpochID %in% epochs[["EpochID"]]) %>% getMeasurements()
#     metaData <- tibble(AllEpochs=NumberAllEpochs,AllSignals=numberAllSignals,EpochsWithReceiverFix=NROW(epochs),SignalsWithReceiverFix=NROW(measurements),ReceivedSignalsWithReceiverFix=sum(!is.na(measurements[["ReceivedSvTimeNanos"]])))
#     gnssData <- list(measurements=measurements,epochs=epochs,metaData=metaData)
#   return(gnssData)

# gnssanalysis matlab source suggests various checks for timenanos (hw clock) fullbiasnanos (week number), potnetnailly bias nanos but zero if mssing, and ditto hqrdware discontinuity count

def read_gnsslogger(filename: str) -> gpd.DataFrame:
    """processs a gnsslogger log file (csv) and returns a set of gnss observations.

    Parameters
    ----------
    filename : str
        filename (with filepath) of the log file.

    Returns
    -------
    gpd.DataFrame
        geopandas dataframe of gnss observations including:
            receiver position (as point geometry)
            observation time
            svid
            signal features e.g. CNO, pr
            , including TimeNanos (int64), FullBiasNanos (int64),Svid,ReceivedSvTimeNanos (int64),PseudorangeRateMetersPerSecond
            and data fields created by this function:
            .allrxMillis (int64), full cycle time of measurement (milliseconds)
            accurate to one millisecond, it is convenient for matching up time
            tags. For computing accurate location, etc, you must still use
            TimeNanos and gnssMeas.trxSeconds
    """
    gnss_raw, gnss_fix = read_csv_(filename)
    gnss_measurements = process_raw(gnss_raw)
    gnss_received_observations = calculate_receiver_position(
        gnss_measurements, gnss_fix)


def read_csv_(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the log file created by Google's gnsslogger Android app.

    Compatible with gnss logger version 2.0.

    Parameters
    ----------
    filename : str
        filename (with filepath) of the log file.

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame]:
        gnss_raw,gnss_fix

        gnss_raw
            all Raw (GnssClock and GnssMeasurement) observations from log file

        gnss_fix
            all Fix (position solutions from Android phone) observations from log file, including:
                Latitude,Longitude,Altitude,UTC Time and time nanos
    """

    # HEADERS<- list(Raw=c("ElapsedRealtimeNanos","TimeNanos","FullBiasNanos","BiasNanos","BiasUncertaintyNanos","Svid","TimeOffsetNanos","State","ReceivedSvTimeNanos","ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond","Constellation","UserLatitude","UserLongitude","UserAltitude"),
    #                 Fix=c("Latitude","Longitude","Altitude","UTCTimeInMs","ElapsedRealtimeNanos"))
    # COLUMNS <- list(Raw=c(2,3,6,7,8,12,13,14,15,16,17,18,29,32,33,34),Fix=c(3,4,5,8,9))

    version = ""
    platform = ""
    minimum_version = "1.4.0.0"

    with open(filename, 'r') as f:
        for line in f:
            if "version" in line.lower():
                start = line.lower().find("version")+9
                end = start+8
                version = line[start:].split(" ", maxsplit=1)[0]
            if "platform" in line.lower():
                start = line.lower().find("platform")+10
                end = start+2
                platform = line[start:end]

            if version != "" and platform != "":
                break

    if platform != "N":
        warnings.warn(
            'Platform not found in log file. Expected "Platform: N".')

    if not _compare_version(version, minimum_version):
        raise ValueError(
            f'Version {version} found in log file. Gnssmapper supports gnsslogger v1.4.0.0 onwards')

    with open(filename, 'r') as f:
        raw = (l.split(",", maxsplit=1)[1].replace(
            " ", "") for l in f if "raw" in l.lower())
        gnss_raw = pd.read_csv("\n".join(raw))

    with open(filename, 'r') as f:
        fix = (l.split(",", maxsplit=1)[1].replace(
            " ", "") for l in f if "fix" in l.lower())
        gnss_fix = pd.read_csv("\n".join(fix))

    return (gnss_raw, gnss_fix)


def _compare_version(actual: str, expected: str) -> bool:
    """Tests whether the log file version meets dependencies"""
    def str_to_list(x):
        l = [int(n) for n in x.split('.')]
        l += [0] * (4 - len(l))
        return l

    for a, e in zip(str_to_list(actual), str_to_list(expected)):
        if a > e:
            return True
        if a < e:
            return False

    return True


def process_raw(gnss_raw: pd.DataFrame) -> pd.DataFrame:
    """Generates signal features from raw measurements.

    Parameters
    ----------
    gnss_raw : pd.DataFrame
        all Raw (GnssClock and GnssMeasurement) observations from log file

    Returns
    -------
    pd.DataFrame
        gnss_raw plus svid_full, rx, tx
    """
    # reformat svid to standard (IGS) format
    # GPS,GLONASS,BEIDOU,GALILEO
    CONSTELLATION = {1: "G", 3: "R", 5: "C", 6: "E"}
    constellation = gnss_raw['Constellation'].map(CONSTELLATION)
    svid_string = gnss_raw['Svid'].astype("string").pad(2, fillchar='0')
    svid_full = constellation.str.cat(svid_string)

    # compute rx (nanos since gps epoch)
    # gnss_raw['BiasNanos','TimeOffsetNanos] can provide subnanosecond accuracy
    rx = gnss_raw['TimeNanos'] - gnss_raw['FullBiasNanos']

    # compute transmission time (nanos since gps epoch)
    tx = gnss_raw['ReceivedSvTimeNanos'] - np.where(
        constellation == 'E', galileo_ambiguity(gnss_raw['ReceivedSvTimeNanos']), 0)
    + np.vectorize(system_time)(rx, gnss_raw['state'], constellation)
    + constellation.map(constants.epoch_offset)

    # This will fail if the rx and tx are in seperate weeks
    # add code to remove a week if failed
    if np.any(rx < tx):
        warnings.warn(
            "rx less than tx, corrected assuming due to different gps weeks")
        tx -= np.where(rx < tx,
                       constellation.map(constants.nanos_in_period), 0)

    # check we have no nonsense psuedoranges
    assert 0 < rx - tx < 1e9, 'Calculated pr time outside 0 to 1 seconds'

    # Pseudorange
    pr = (rx-tx) * 1e-9 * constants.lightspeed

    # UTC time
    utc_rx = gps_to_utc(rx)

    return pd.concat([gnss_raw, svid_full, rx, tx, utc_rx, pr])


def galileo_ambiguity(x: int) -> int:
    """ Galileo measurements may collapse into measuring pilot stage with 100ms ambiguity """
    return constants.nanos_in_period['E'] * np.floor(x / constants.nanos_in_period['E'])


def system_time(rx: int, state: int, constellation: str) -> int:
    """Calculates the start time for the gnss system period (since gps epoch in nanoseconds)"""
    required_states = {
        'G': [1, 8],
        'R': [1, 8, 128],
        'C': [1, 8],
        'E': [1, 8, 2048],
    }

    def start_of_period(x): return constants.nanos_in_period[constellation] * floor(
        x / constants.nanos_in_period[constellation])

    if all(state & n for n in required_states[constellation]):
        return start_of_period(rx)
    else:
        return np.nan


def gps_to_utc(time: int) -> np.array:
    """ converts nanoseconds since gps epoch into a utc time"""
    utc = pd.to_datetime(time, units='ns', origin=constants.gps_epoch)
    ls_end = constants.leap_seconds(utc)
    # using the leap_seconds at start to avoid a roll over situation.
    ls_start = constants.leap_seconds(utc - ls_end)

    return utc - ls_start


# getMeasurements <- function(rawData){
#   epochIDs <- unique(rawData[["EpochID"]])
#   svids <- c(c("E01","E02","E03","E04","E05","E06","E07","E08","E09","E10","E11","E12","E13","E14","E15","E16","E17","E18","E19","E20","E21","E22","E23","E24","E25","E26","E27","E28","E29","E30","E31","E32","E33","E34","E35","E36"),
#             c("G01","G02","G03","G04","G05","G06","G07","G08","G09","G10","G11","G12","G13","G14","G15","G16","G17","G18","G19","G20","G21","G22","G23","G24","G25","G26","G27","G28","G29","G30","G31","G32"),
#             c("R01","R02","R03","R04","R05","R06","R07","R08","R09","R10","R11","R12","R13","R14","R15","R16","R17","R18","R19","R20","R21","R22","R23","R24"),
#             c("C01","C02","C03","C04","C05","C06","C07","C08","C09","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","C27","C28","C29","C30","C31","C32","C33","C34","C35","C36","C37"))
#   measurements <- tibble(EpochID = rep(epochIDs,each=length(svids)),
#                    Svid = rep(svids,times=length(epochIDs)))

#   measurements <- left_join(measurements,rawData,by=c("EpochID","Svid")) %>% dplyr::select(-UserLatitude,-UserLongitude,-UserAltitude,-SystemTime,-UTCTime)
#   drop <- nrow(rawData)-sum(!is.na(measurements[["ReceivedSvTimeNanos"]]))

#   cat(sum(!is.na(measurements[["ReceivedSvTimeNanos"]])),"signals received out of a potential",nrow(measurements),"\n",
#       drop," additional signals were received but could not be matched to our svid database (covers GPS,Galileo,Glonass and Beidou)\n"
#   )

#   return(measurements)
# }

# getEpochs <- function(gnssData,onlyMarkedLocations=TRUE,LocationList=NULL,useFixes=FALSE){
#   #This function produces a data table of time and location for each epoch. If the location is provided by the user it is used.
#   #userLocationList indicates whether the LLA has been fully specified or whether a location number has been provided (in Latitude column) to be looked up iun a list.
#   #if usesFixes is true, epochs location will be estimated based on the GNSS reciever measurements.

#   epochs <- distinct(gnssData[["Raw"]],EpochID, .keep_all=TRUE) %>% dplyr::select(EpochID,SystemTime,UTCTime,Latitude=UserLatitude,Longitude=UserLongitude,Altitude=UserAltitude)
#   if(!is.null(LocationList)){
#     epochs <- writeUserLLA(epochs,LocationList)
#   }
#   if (useFixes){
#     epochs[!missing(epochs),] <- getDeviceFixes(gnssData[["Fix"]],epochs[!missing(epochs),])
#     if(!onlyMarkedLocations){epochs[missing(epochs),] <- getDeviceFixes(gnssData[["Fix"]],epochs[missing(epochs),])}

#   }

# return(epochs)
# }
