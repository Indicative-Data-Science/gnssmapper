User Guide
===========

Introduction
------------
The potential exists to build 3D maps using Global Navigation Satellite System (GNSS) data.
3D maps are increasingly useful for analysing and planning cities, location-based services, and autonomous vehicle and drone navigation. GNSS data could be a low-cost replacement for existing map-creation methods, such as photogrammetry and LIDAR. 

GnssMapper is a Python package for the entire workstream of building a 3D map from GNSS data. It aspires to be useful to researchers in the field of 3D mapping and to laypersons interested in exploring such techniques. The first aim is to provides access to GNSS data without requiring any in-depth understanding of GNSS concepts. The second aim is to automate the workflow of processing GNSS data into a format usable by mapping algorithms. The third aim is to provide a framework for specifying mapping algorithms.

What is GNSS?
^^^^^^^^^^^^^
A Global Navigation Satellite System (GNSS) is a navigation system which relies on satellites orbiting the earth and broadcasting signals.
A GNSS receiver can use these signals to calculate its position and time.

More precisely, the GNSS signal comprises a ranging code, also known as the pseudo-random-noise (PRN) code, and navigation data. 
The receiver uses the ranging code to identify the transmitting satellite and calculate the signal travel time from satellite to receiver, which is multiplied by the speed of light to provide the apparent distance, known as the pseudorange. 
The navigation data provides information on the satellite location along with supplementary information. 
Knowledge of satellite locations and pseudoranges allows the receiver to calculate its position and time if enough satellites (usually 4 or more) are received. 

Determining location from GNSS data can be relatively complex. Rather than implement methods, GnssMapper assumes your gnss data has already been processed to generate position estimates. 
In the first place, gnss receivers have on-board software for generating position solutions that should be satisfactory for most purposes.
Furthermore there are a number of existing open source packages available that implement different processing methods.

*   For Python, `Laika <https://github.com/commaai/laika>`_ is a simple GNSS processing library. 
*   More options can be found on a useful `list of software, data, and resources <https://github.com/barbeau/awesome-gnss>`_ maintained by Sean Barbeau at the University of South Florida.
*   If different approaches are of interest, guides to the principles of a GNSS system are widely available, for example `Using GNSS raw measurements on mobile devices <https://www.gsa.europa.eu/system/files/reports/gnss_raw_measurement_web_0.pdf>`_ by the European GNSS Agency, or `Gnss data processing. volume 1: Fundamentals and algorithms <https://gssc.esa.int/navipedia/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf>`_ by the European Space Agency.

Instead, GnssMapper uses the positions of the satellites and receivers along with observed features of the satellite signals (primarily signal strength) to generate a 3D map.

What's the difference between GPS and GNSS?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GPS is an instance of GNSS. It was the first GNSS to be developed, by the USA, however other GNSS constellations have since been developed by other countries: the major ones are Beidou (China), Galileo (EU) and Glonass (Russia).

Depending on your receiver, you will be able to pick up different constellations, with it becoming more and more common for multiple constellations to be supported.


Geospatial Processing
---------------------
The tools need to process geospatial data already exist within the Python environment (GeoPandas, Fiona, PyProj, Shapely) and GnssMapper is built on top of them. GnssMapper aims to provide a small set of operations specific to GNSS mapping, in an optimised and robust way, suitable for novices.


Data Model
----------
The fundamental types of object in GnssMapper are ReceiverPoints, Rays, Observations, and Maps.

* A ReceiverPoint is a Point in three dimensions of space plus time. It represents a Receiver's location along with any additional recorded signal features from received GNSS signals.
* A Ray is a single segment 3D Linestring representing a direct path from a receiver to a satellite. It is truncated in order to avoid problems when changing coordinate reference systems. 
* An Observation is a set of Rays along with signal features relating to the signals
* A Map is a collection of 2D polygons along with heights, which represent a 2.5D (otherwise known as 3D LOD 1) map. This is where all 3D feautres are considered to be extruded blocks from ground level. 

The implementation of these objects are as GeoPandas GeoDataFrames (a GeoSeries in the case of a Ray).
This allows access to all existing methods for geospatial data. This also means its possible to alter a GnssMapper object so that it's no longer valid.
This is dealt with through verification checking where needed.


Coordinate Systems
------------------
2D maps are usually in a projected coordinate system, that transforms the spherical earth into a flat plane. To add height to a map, a vertical reference system is needed. If a map does not have an associated vertical CRS, it is assumed that heights are in relation to the WGS84 ellipsoid.

GNSS locations (of receivers and satellites) are well defined in 3D and e reported in WGS84 format.

Support for transformations of 3D CRS varies. It may be the case that the base transformation does not proivde the expected result. It is recommended that careful checking is made of the transformation and that `PyProj network settings <https://pyproj4.github.io/pyproj/stable/api/network.html>`_ are enabled, to allow the best possible transformation to be used. pyproj can be accessed through gnssmapper.geo.pyproj


Time
----
The timestamp associated with gnss data varies between constellation systems. GnssMapper always assumes times are provided in UTC. 


Constructing objects
--------------------
Objects can be created by any GeoPandas method, as long as the object passes the verification checks.

GNSS Data
-------------
GnssMapper can parse raw data generated by the GnssLogger app. It doesn't have the ability to parse RINEX files, altohugh the parsed output can be used as Receiverpoints if in the correct format.




