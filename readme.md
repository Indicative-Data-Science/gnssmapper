#**GnssMapper**
===========

Tools for generating 3D maps from GNSS data

Introduction
-----------
GnssMapper provides tools for generating 3D maps by using Global Navigation Satellite System (GNSS) data. It is connected to a [research project](https://indicative-data-science.github.io/IDS/) at the University of Glasgow, which investigates methods for using crowdsourced GNSS data for mapping. It is written in Python and built upon [GeoPandas](https://geopandas.org) objects

It provides the following capabilities:
- read data from the [gnsslogger](https://github.com/google/gps-measurement-tools) app, available for Android phones.
- process data into a set of observations
- use algorithms to estimate building heights
- simulate observations for algorithm testing

Install
-------
GnssMapper depends on the GeoPandas package and we suggest following the recommended instructions for installing that package.

GnssMapper runs on Python 3.6 and above.

Examples
--------
Most methods return GeoPandas GeoDataFrame in particular forms.

The receiverpoints form is a collection of points in 3D, with a time column, as well as additional signal features (e.g. satellite identifier (svid) and signal strength(CN0DbHz)) identifying the position in space and time of the receiver which collected the GNSS data. Positions can be repeated as each row corresponds to a single satellite observation for that epoch.  

    >>> import gnssmapper
    >>> l = gnssmapper.read_gnsslogger('./example/example.txt')
    >>> l

The observations form is a collection of rays in 3D, along with signal features. A ray is a single segment linestring which represents the LOS path from receiver to satellite. In order to retrieve information on the historic positions of satellites, GnssMapper downloads data from the ESA. Downloading and parsing the data is slow, so a local cache is generated, and loaded into memory as required.  




The map form is a collection of 2D geometries, with a height column. This represents a simple LOD1 3D map. It can be initialised from a 2D map with a blank height column. 











Example Data
------------
In the folder 'examplefiles' there are a series of gnsslogger files created as part of a pilot study, that can be used for testing and analysis.









