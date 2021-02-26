# **GnssMapper**

Tools for generating 3D maps from GNSS data

## Introduction
-----------
GnssMapper provides tools for generating 3D maps by using Global Navigation Satellite System (GNSS) data. It is connected to a [research project](https://indicative-data-science.github.io/IDS/) at the University of Glasgow, which investigates methods for using crowdsourced GNSS data for mapping. It is written in Python and built upon [GeoPandas](https://geopandas.org) objects.

It provides the following capabilities:
- read 'raw' GNSS data from Google's [gnsslogger](https://github.com/google/gps-measurement-tools) app, available for Android phones
- process data into a set of observations
- estimate building heights based on the observations
- simulate observations for algorithm testing

It does not include any functionality for processing GNSS data in order to estimate position, and assumes position data is available from the log file, or calculated elsewhere. 

## Installation
-------
GnssMapper depends on the GeoPandas package (and underlying dependencies for pandas, shapely, fiona, and pyproj). It also depends on pygeos to speed up vectorised geometry operations. We recommend following the Geopandas instructions for installing these packages.

GnssMapper runs on Python 3.6 and above. GnssMapper has only been tested with the following setup:  
GEOS       : 3.9.0  
GDAL       : 3.2.1  
PROJ       : 7.2.1  
  
geopandas  : 0.8.2  
pandas     : 1.2.2  
fiona      : 1.8.18  
numpy      : 1.19.5  
shapely    : 1.7.1  
pyproj     : 3.0.0.post1  
pygeos     : 0.9  


## Examples
--------
Most methods return GeoPandas GeoDataFrames in particular forms.


### Receiverpoints

A set of GNSS data generated from gnsslogger output. A collection of 3D points with time column, representing receiver position, along with additional signal features (e.g. satellite identifier (svid) and signal strength(CN0DbHz)). An example gnsslogger file is included in the folder 'examplefiles'.

    >>> import gnssmapper as gm
    >>> log = gm.read_gnsslogger("./examplefiles/gnss_log_2020_02_11_08_49_29.txt")
    >>> log[['svid','time','Cn0DbHz','geometry']].head()
      svid                          time    Cn0DbHz                               geometry
    0  G02 2020-02-11 08:49:27.999559028   22.34062  POINT Z (-0.13414 51.52471 114.85894)
    1  G05 2020-02-11 08:49:27.999559028  26.320181  POINT Z (-0.13414 51.52471 114.85894)
    2  G07 2020-02-11 08:49:27.999559028  47.322662  POINT Z (-0.13414 51.52471 114.85894)
    3  G09 2020-02-11 08:49:27.999559028  35.282738  POINT Z (-0.13414 51.52471 114.85894)
    4  G13 2020-02-11 08:49:27.999559028  22.712795  POINT Z (-0.13414 51.52471 114.85894)

### Observations
Observations are a processed set of GNSS data for use in the mapping algorithm, generated from Receiverpoints. . A collection of 3D rays in 3D, along with signal features. A ray is a single segment linestring which represents a direct path from the receiver towards the relevant satellite, truncated at 1km in length. In order to retrieve information on the historic positions of satellites, GnssMapper downloads data from the ESA. Downloading and parsing the data is slow, so a local cache is generated, and loaded into memory as required.  

    >>> obs = gm.observe(pilot_log)
    {'2020063', '2020045', '2020066', '2020044'} orbits are missing and must be created.
    downloading sp3 file for 2020063.
    creating 2020063 orbit.
    saving 2020063 orbit.
    ....
    >>> obs.head()
                      time svid  Cn0DbHz                                           geometry
    0  2020-03-03T10:20:19  C10      NaN  LINESTRING Z (3976545.346 -9309.219 4970128.21...
    1  2020-03-03T10:20:19  C14      NaN  LINESTRING Z (3976545.346 -9309.219 4970128.21...
    2  2020-03-03T10:20:19  C21      NaN  LINESTRING Z (3976545.346 -9309.219 4970128.21...
    3  2020-03-03T10:20:19  C22      NaN  LINESTRING Z (3976545.346 -9309.219 4970128.21...
    4  2020-03-03T10:20:19  C24      NaN  LINESTRING Z (3976545.346 -9309.219 4970128.21...

### Mapping Algorithm
The expected map form is a collection of 2D polygons, with a height column. This represents a simple LOD1 3D map. It can be initialised from a 2D map with a blank height column. For the pilot study, a map of a single building was generated from Ordnance Survey's Mastermap.   

    >>> map_ = gpd.read_file('./examplefiles/map.geojson')
    >>> map_
       height                                           geometry
    0       0  POLYGON ((529552.750 182350.500, 529548.950 18...

Given a map of floorplates and a set of observations, the height of map elements can be predicted from the observations. GnssMapper implements a bootstrapped four-parameter logistic regression developed by the project. This fits a four-parameter logistic regression to the data and estimates the height based on model parameters.

    >>> gm.predict(map_,obs)
       lower_bound  mid_point  upper_bound
    0    47.359955  52.545442     57.73093

![Example 1](docs/fit.png)

### Simulation
In order to test the mapping algorithm, GnssMapper includes the ability to simulate observations. This uses a map (with a ground truth height) and generates a set of Observations by simulating the Receiver location and generating a signal strength based on fresnel attenuation of the rays. 

    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> start = pd.Timestamp('2020-02-11T11')
    >>> end = pd.Timestamp('2020-02-11T12')
    >>> sim = gm.simulate(map_, "point_process", 100, start, end)
    >>> sim.head()
                               time svid                                           geometry     fresnel    Cn0DbHz
    0 2020-02-11 11:49:20.360557432  C10  LINESTRING Z (529644.220 182254.036 1.000, 530...         0.0  34.165532
    1 2020-02-11 11:49:20.360557432  C14  LINESTRING Z (529644.220 182254.036 1.000, 528...  116.001472       <NA>
    2 2020-02-11 11:49:20.360557432  C21  LINESTRING Z (529644.220 182254.036 1.000, 529...         0.0  39.337049
    3 2020-02-11 11:49:20.360557432  C24  LINESTRING Z (529644.220 182254.036 1.000, 528...   96.973759       <NA>
    4 2020-02-11 11:49:20.360557432  C26  LINESTRING Z (529644.220 182254.036 1.000, 529...   59.631021       <NA>

## Example Data
----------
In the folder 'examplefiles' there is a receiverpoint file created as part of a pilot study, that can be used for testing and analysis. This can be loaded using GeoPandas but note that some processing of datatypes is required

    >>> import geopandas as gpd
    >>> pilot_log = gpd.read_file("zip://./examplefiles/pilot_study.geojson.zip", driver="GeoJSON")
    >>> pilot_log.time = pilot_log.time.astype('datetime64')
    >>> pilot_log.svid = pilot_log.svid.astype('string')

## Useful GNSS references
--------
- [Laika, a simple Python GNSS processing library](https://github.com/commaai/laika)
- [Overview of GNSS concepts and raw data](https://www.gsa.europa.eu/system/files/reports/gnss_raw_measurement_web_0.pdf)










