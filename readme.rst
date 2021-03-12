==========
GnssMapper
==========

Tools for generating 3D maps from GNSS data

Introduction
============
GnssMapper provides tools for generating 3D maps by using Global Navigation Satellite System (GNSS) data. It is connected to a `research project <https://indicative-data-science.github.io/IDS/>`__ at the University of Glasgow, which investigates methods for using crowdsourced GNSS data for mapping. It is written in Python and built upon `GeoPandas <https://geopandas.org>`__ objects.

It provides the following capabilities:

* read 'raw' GNSS data from Google's `gnsslogger <https://github.com/google/gps-measurement-tool>`__ app, available for Android phones
* process data into a set of observations
* estimate building heights based on the observations
* simulate observations for algorithm testing

It does not include any functionality for processing GNSS data in order to estimate position, and assumes position data is available from the log file, or calculated elsewhere. 

For more details see:
  
* `GnssMapper github repository <https://github.com/Indicative-Data-Science/gnssmapper>`__
* `GnssMapper documentation <https://gnssmapper.readthedocs.io/>`__

Requirements
============
GnssMapper depends on the GeoPandas package and its underlying dependencies, including PyGeos. If you do not have these installed, we recommend following the `instructions <https://gnssmapper.readthedocs.io/en/latest/getting_started/installation.html>`__.

GnssMapper has only been tested with the following setup:  
    Python     : 3.9.1
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

Installation
============
Distribution are available from the Python Package Index

.. code-block:: console

    $ pip install gnssmapper


Get in touch
============
Report bugs, suggest features or view the source code at https://github.com/Indicative-Data-Science/gnssmapper.

Examples
========
Most methods return GeoPandas GeoDataFrames in particular forms.

Receiverpoints
--------------

A set of GNSS data generated from GnssLogger output. A collection of 3D points with time column, representing receiver position, along with additional signal features.
.. code-block:: pycon

    >>> import gnssmapper as gm
    >>> log = gm.read_gnsslogger("./examplefiles/gnss_log_2020_02_11_08_49_29.txt")
    >>> log[['svid','time','Cn0DbHz','geometry']].head()
      svid                          time    Cn0DbHz                               geometry
    0  G02 2020-02-11 08:49:27.999559028   22.34062  POINT Z (-0.13414 51.52471 114.85894)
    1  G05 2020-02-11 08:49:27.999559028  26.320181  POINT Z (-0.13414 51.52471 114.85894)
    2  G07 2020-02-11 08:49:27.999559028  47.322662  POINT Z (-0.13414 51.52471 114.85894)
    3  G09 2020-02-11 08:49:27.999559028  35.282738  POINT Z (-0.13414 51.52471 114.85894)
    4  G13 2020-02-11 08:49:27.999559028  22.712795  POINT Z (-0.13414 51.52471 114.85894)

Observations
------------
Processed GNSS data for use in the mapping algorithm. 
Each observation is a single segment linestring from the receiver towards the relevant satellite, along with signal features. 
.. code-block:: pycon

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

Maps
----
The map form is a collection of 2D polygons, with a height column. This represents a simple LOD1 3D map. It can be initialised from a 2D map with a blank height column::
.. code-block:: pycon

    >>> map_ = gpd.read_file('./examplefiles/map.geojson')
    >>> map_
       height                                           geometry
    0       0  POLYGON ((529552.750 182350.500, 529548.950 18...

Given a map of floorplates and a set of observations, the height of map elements can be predicted from the observations::
.. code-block:: pycon

    >>> gm.predict(map_,obs)
       lower_bound  mid_point  upper_bound
    0    47.359955  52.545442     57.73093

Simulation
----------
GnssMapper can simulate observations if given a map, based on fresnel attenuation of the rays. 
.. code-block:: pycon

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

Example Data
------------
https://github.com/Indicative-Data-Science/gnssmapper/tree/master/examplefiles has an example gnsslogger file and a receiverpoint file created as part of a pilot study, that can be used for testing and analysis. This can be loaded using GeoPandas but note that some processing of datatypes is required
.. code-block:: pycon

    >>> pilot_log = gpd.read_file("zip://./examplefiles/pilot_study.geojson.zip", driver="GeoJSON")
    >>> import geopandas as gpd
    >>> pilot_log.time = pilot_log.time.astype('datetime64')
    >>> pilot_log.svid = pilot_log.svid.astype('string')


    









