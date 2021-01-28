""" 
Module containing methods for processing raw GNSS data as produced by Android phones by e.g. gnss logger.
"""

import pandas as pd
from typing import Tuple
import warnings

# readLog <- function(file,calculatePR,...){
#     cat("Reading",file,"\n")
#     data <- readGnss(file) %>% preprocessGnss(calculatePR)
#     epochs <- getEpochs(data,...) 
#     NumberAllEpochs <- NROW(epochs)
#     numberAllSignals <- NROW(data[["Raw"]])
#     epochs <- epochs[!missing(epochs) & !is.na(epochs$UTCTime),] #ensures valid time and position measurement
#     measurements <- data[["Raw"]] %>% filter(EpochID %in% epochs[["EpochID"]]) %>% getMeasurements()
#     metaData <- tibble(AllEpochs=NumberAllEpochs,AllSignals=numberAllSignals,EpochsWithReceiverFix=NROW(epochs),SignalsWithReceiverFix=NROW(measurements),ReceivedSignalsWithReceiverFix=sum(!is.na(measurements[["ReceivedSvTimeNanos"]])))
#     gnssData <- list(measurements=measurements,epochs=epochs,metaData=metaData)
#   return(gnssData)

### gnssanalysis matlab source suggests various checks for timenanos (hw clock) fullbiasnanos (week number), potnetnailly bias nanos but zero if mssing, and ditto hqrdware discontinuity count



def read_gnsslogger(filename:str)-> Tuple[pd.DataFrame,pd.DataFrame]:
    """Reads the log file created by Google's gnss logger Android app.

    Compatible with gnss logger version 2.0.

    Parameters
    ----------
    filename : str
        filename (with filepath) to the log file. 

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame]
        gnssRaw,gnssFix
        
        gnssRaw
            all Raw (GnssClock and GnssMeasurement) observations from log file, including TimeNanos (int64), FullBiasNanos (int64),Svid,ReceivedSvTimeNanos (int64),PseudorangeRateMetersPerSecond
            and data fields created by this function:
            .allRxMillis (int64), full cycle time of measurement (milliseconds)
            accurate to one millisecond, it is convenient for matching up time 
            tags. For computing accurate location, etc, you must still use 
            TimeNanos and gnssMeas.tRxSeconds

        gnssFix
            all Fix (position solutions from Android phone) observations from log file, including:
                Latitude,Longitude,Altitude,UTC Time and time nanos
    """  

    # HEADERS<- list(Raw=c("ElapsedRealtimeNanos","TimeNanos","FullBiasNanos","BiasNanos","BiasUncertaintyNanos","Svid","TimeOffsetNanos","State","ReceivedSvTimeNanos","ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond","Constellation","UserLatitude","UserLongitude","UserAltitude"),
    #                 Fix=c("Latitude","Longitude","Altitude","UTCTimeInMs","ElapsedRealtimeNanos"))
    # COLUMNS <- list(Raw=c(2,3,6,7,8,12,13,14,15,16,17,18,29,32,33,34),Fix=c(3,4,5,8,9))

    version=""
    platform=""
    minimum_version="1.4.0.0"

    with open(filename,'r') as f:
        for line in f:
            if "version" in line.lower():
                start=line.lower().find("version")+9
                end=start+8
                version=line[start:].split(" ",maxsplit=1)[0]
            if "platform" in line.lower():
                start=line.lower().find("platform")+10
                end=start+2
                platform=line[start:end]
                        
            if version!="" and platform!="":
                break
        
    if platform!="N":
        warnings.warn('Platform not found in log file. Expected "Platform: N".')
    
    if not _compare_version(version,minimum_version):
        raise ValueError(f'Version {version} found in log file. Gnssmapper supports gnsslogger v1.4.0.0 onwards')

    with open(filename,'r') as f:
        raw=(l.split(",",maxsplit=1)[1].replace(" ","") for l in f if "raw" in l.lower())
        gnss_raw=pd.read_csv("\n".join(raw))

    with open(filename,'r') as f:
        fix=(l.split(",",maxsplit=1)[1].replace(" ","") for l in f if "fix" in l.lower())
        gnss_fix=pd.read_csv("\n".join(fix))

    return (gnss_raw,gnss_fix)

    
    
def _compare_version(actual:str,expected:str)->bool:
    """Tests whether the log file version meets dependencies"""
    def str_to_list(x):
        l=int(n) for n in x.split('.')
        l+=[0]*(4-len(l))
        return l

    for a,e in zip(str_to_list(actual),str_to_list(expected)):
        if a>e: return True
        if a<e: return False
    
    return True


def process_raw_gnss()
# preprocessGnss <- function(rawData,calculatePR){
#   CONSTELLATION <- c("G"=1,"R"=3,"C"=5,"E"=6) #GPS,GLONASS,BEIDOU,GALILEO
#   GPS_UTC_OFFSET <- as.integer(as.POSIXct('1980-01-06',tz="UTC"))
#   GPS_UTC_LEAPSECONDS <- -18
#   epochTimes <- unique(rawData[["Raw"]][["ElapsedRealtimeNanos"]]/1e9)
  
#   Raw <- 
#     rawData[["Raw"]] %>% 
#     mutate(Svid = paste0(names(CONSTELLATION)[match(Constellation,CONSTELLATION)],formatC(Svid,width=2,flag="0")),
#            UTCTime = as.POSIXct(as.integer((TimeNanos-(FullBiasNanos+BiasNanos))/1e9) +GPS_UTC_OFFSET+GPS_UTC_LEAPSECONDS,origin = "1970-01-01", tz = "UTC"),
#            ReceiverGPSTimeNanos = ifelse(FullBiasNanos=="",NA_real_,TimeNanos-(FullBiasNanos+BiasNanos)+TimeOffsetNanos), #This checks for a valid time fix for the epoch
#            TransmitterGPSTimeNanos = ifelse(calculatePR,calculateTransmitterTimeNanos(ReceivedSvTimeNanos,names(CONSTELLATION)[match(Constellation,CONSTELLATION)],State,ReceiverGPSTimeNanos),NA_real_), #this converts the timestamp of the sv transmission into a GPS time format, if ambiguities are resolvable
#            SystemTime=ElapsedRealtimeNanos/1e9,
#            EpochID=match(SystemTime,epochTimes),
#            UserLongitude=ifelse(is.nan(UserLongitude),NA,UserLongitude),
#            UserAltitude=ifelse(is.nan(UserAltitude),NA,UserAltitude)) %>% 
#     dplyr::select(-TimeNanos,-TimeOffsetNanos,-ElapsedRealtimeNanos,-FullBiasNanos,-BiasNanos,-Constellation) 
  
#   if(is_null(rawData[["Fix"]])){
#     Fix <- rawData[["Fix"]]
#   } else {
#     Fix <- 
#       rawData[["Fix"]] %>% 
#       mutate(UTCTime = as.POSIXct(UTCTimeInMs/1e3,origin = "1970-01-01", tz = "UTC"),
#              SystemTime=ElapsedRealtimeNanos/1e9) %>%
#       dplyr::select(SystemTime,UTCTime,Latitude,Longitude,Altitude) 
#   }  
#   return(list(Raw=Raw,Fix=Fix))
# }


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


# stateCheck <- function(state,number) {
#   asReverseBinary <- function(x){
#     if (x<=1){
#       return (x %% 2)
#     } else {
#       return(c(x %% 2,asReverseBinary  (x %/% 2)))
#     }
#   }
  
#   bin <- asReverseBinary(state)
#   test <- length(asReverseBinary(number))
#   return(length(bin)>=test && bin[test]==1)  
  
# }
# readGnss <- function(filename){
#   #Reads a log file and returns 2 dataframes: raw signals, and associated epoch)
#   DATA_TYPES <- c(Raw="Raw",Fix="Fix")
#   HEADERS<- list(Raw=c("ElapsedRealtimeNanos","TimeNanos","FullBiasNanos","BiasNanos","BiasUncertaintyNanos","Svid","TimeOffsetNanos","State","ReceivedSvTimeNanos","ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond","Constellation","UserLatitude","UserLongitude","UserAltitude"),
#                  Fix=c("Latitude","Longitude","Altitude","UTCTimeInMs","ElapsedRealtimeNanos"))
#   COLUMNS <- list(Raw=c(2,3,6,7,8,12,13,14,15,16,17,18,29,32,33,34),Fix=c(3,4,5,8,9))
  
#   extractData <- function(data,type){
#     a <- data[startsWith(data,type)]
#     if(length(a)==0) {
#       return(NULL)
#     } else {
#       b <- read_csv(file=a,col_names = FALSE)
#       c <- b[COLUMNS[[type]]]
#       colnames(c) <- HEADERS[[type]]
#       return(c)
#     }
#   } 
  
#   log <- readLines(filename)
#   data <- map(DATA_TYPES,~extractData(log,.))
#   return(data)
# }

# calculateTransmitterTimeNanos <- function(ReceivedSvTimeNanos,ConstellationLetter,State,ReceiverGPSTimeNanos){
#   calculateGPSTravelTime <- function(ReceivedSvTimeNanos,State,ReceiverGPSTimeNanos) {
#     if(!stateCheck(State,8)) {return(NA_real_)} #requiring full TOW with no measurement ambiguity. 
#     NUMBER_NANOS_WEEK <-604800*10^9 
#     ReceiverTimeNanos <- ReceiverGPSTimeNanos - NUMBER_NANOS_WEEK * (ReceiverGPSTimeNanos %/% NUMBER_NANOS_WEEK)
    
#     return (ReceiverGPSTimeNanos-(ReceiverTimeNanos-ReceivedSvTimeNanos))
#   }
 
#   calculateGalileoTravelTime <- function(ReceivedSvTimeNanos,State,ReceiverGPSTimeNanos) {
#     if(!stateCheck(State,8)| !stateCheck(State,2048) ) {return(NA_real_)} #requiring second code lock as well as TOW decoded.
#     NUMBER_NANOS_100MILLIS <-10^8 
#     ReceiverTimeNanos <- ReceiverGPSTimeNanos - NUMBER_NANOS_100MILLIS * (ReceiverGPSTimeNanos %/% NUMBER_NANOS_100MILLIS)
#     ReceivedSvPilotTimeNanos <- ReceivedSvTimeNanos - NUMBER_NANOS_100MILLIS * (ReceivedSvTimeNanos %/% NUMBER_NANOS_100MILLIS) #Documentation suggests that the measurement can collapse into measuring pilot stage with 100ms ambiguity. This step ensures conformity of measurment
#     return (ReceiverGPSTimeNanos-(ReceiverTimeNanos-ReceivedSvPilotTimeNanos))
#   } 
  
#   calculateBeidouTravelTime <- function(ReceivedSvTimeNanos,State,ReceiverGPSTimeNanos) {
#     if(!stateCheck(State,8)) {return(NA_real_)} #requiring full TOW with no measurement ambiguity. 
#     NUMBER_NANOS_WEEK <-604800*10^9 
#     BEIDOU_OFFSET <- 14*10^9
#     ReceiverTimeNanos <- ReceiverGPSTimeNanos - NUMBER_NANOS_WEEK * (ReceiverGPSTimeNanos %/% NUMBER_NANOS_WEEK) - BEIDOU_OFFSET
#     return (ReceiverGPSTimeNanos-(ReceiverTimeNanos-ReceivedSvTimeNanos))
#   }
  
#   calculateGlonassTravelTime <- function(ReceivedSvTimeNanos,State,ReceiverGPSTimeNanos) {
#     if(!stateCheck(State,128)) {return(NA_real_)} #requiring full TOD with no measurement ambiguity. 
#     NUMBER_NANOS_DAY <-86400*10^9 
#     GLONASS_3HOUR_OFFSET <- 10800*10^9
#     GLONASS_LEAPSECOND_OFFSET <- -18*10^9
#     ReceiverTimeNanos <- ReceiverGPSTimeNanos - NUMBER_NANOS_DAY * (ReceiverGPSTimeNanos %/% NUMBER_NANOS_DAY) + GLONASS_3HOUR_OFFSET + GLONASS_LEAPSECOND_OFFSET
#     return (ReceiverGPSTimeNanos-(ReceiverTimeNanos-ReceivedSvTimeNanos))
#   }
  
#   TransmitterGPSTimeNanos <- pmap_dbl(list(ConstellationLetter,ReceivedSvTimeNanos,State,ReceiverGPSTimeNanos),~switch(..1,
#                                                                                                           "G"=calculateGPSTravelTime(..2,..3,..4),
#                                                                                                           "R"=calculateGlonassTravelTime(..2,..3,..4),
#                                                                                                           "C"=calculateBeidouTravelTime(..2,..3,..4),
#                                                                                                           "E"=calculateGalileoTravelTime(..2,..3,..4),
#                                                                                                           NA_real_))
#   return(TransmitterGPSTimeNanos)
# }