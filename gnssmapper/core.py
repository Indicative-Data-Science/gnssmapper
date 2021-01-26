""" 
Contains the core of gnssmapper.
A private module. All functions accessible from main namespace

"""

import geopandas as gpd

def check_valid_observations(obs:gpd.GeoDataFrame)->bool:
    """Checks a geodataframe is a valid set of observations."""
    pass

def check_valid_receiverpoints(points:gpd.GeoDataFrame)->bool:
    """Checks a geodataframe is a valid set of receiver points."""
    pass

def read(filename:str)->gpd.GeoDataFrame:
    """Reads a gnsslogger csv file and returns aset of gnss observations.

    Parameters
    ----------
    filename : str
        path to gnsslogger file.

    Returns
    -------
    gpd.GeoDataFrame
        A geopandas dataframe of gnss observations with a specific structure (see observations in Documentation)
    """
    
    obs_no_sat = _read_gnsslogger(filename)
    return locate_satellite(obs_no_sat)  #need to change 
    
def _read_gnsslogger(filename:str)->gpd.GeoDataFrame:
    "reads a gnss logger file and "




