Tutorial
========

This tutorial provides an introduction to the key functionality of GnssMapper. It is a step-by-step guide through the process of working with GNSS data.

Obtaining GNSS data
-------------------
For most people, the best place to obtain GNSS data is from their phone. Most smartphones have internal GNSS receivers, and phones running the Android operating system are able to access the underlying data straightforwardly. Unfortunately Apple phones do not have a similar functionality. The following steps will allow Android users to record data.

1. Obtain `GnssLogger <https://play.google.com/store/apps/details?id=com.google.android.apps.location.gps.gnsslogger>`_ for your phone from the Android Play store. This is a Google developed app for showcasing the ability of using GNSS data.
2. Once installed, open the app and navigate to the Settings tab. Select the toggles for Location and Measurements. This allows the app to record both the signal data and the phone's position estimates (based on the gnss receiver as well as any additional information available to the phone). 
3. Then navigate to the Log tabscreen and click 'Start Log' to begin recording. Once you have enough data, click 'Stop & Send' to save onto your phone or share directly in a csv (comma separated file) format.

Preprocessing GNSS data into input for a mapping algorithm
----------------------------------------------------------

Before applying the data to a mapping algorithm, it must be processed into a suitable form, and GnssMapper provides the necessary tools to be able to do that.
First we parse the output from GNSSLogger into a Receiverpoints format. This is a dataframe of recorded signals with two columns:

*   geometry - a 3D point, representing the receivers position in space
*   time - the time (in UTC) that the signal was received

The dataframe will also include all the various raw data that GNSSLogger outputs (as described in the `Android documentation <https://developer.android.com/guide/topics/sensors/gnss>`_) but for now, lets focus on two key ones:

*   svid - satellite identifier (svid)
*   Cn0DbHz - The Carrier-to-Noise ratio, which is a measure of signal strength.

::

    >>> import gnssmapper as gm
    # Replace the filepath with your saved GNSSLogger file.
    >>> log = gm.read_gnsslogger("./examplefiles/gnss_log_2020_02_11_08_49_29.txt")
    >>> log[['svid','time','Cn0DbHz','geometry']].head()
        svid                          time    Cn0DbHz                               geometry
    0  G02 2020-02-11 08:49:27.999559028   22.34062  POINT Z (-0.13414 51.52471 114.85894)
    1  G05 2020-02-11 08:49:27.999559028  26.320181  POINT Z (-0.13414 51.52471 114.85894)
    2  G07 2020-02-11 08:49:27.999559028  47.322662  POINT Z (-0.13414 51.52471 114.85894)
    3  G09 2020-02-11 08:49:27.999559028  35.282738  POINT Z (-0.13414 51.52471 114.85894)
    4  G13 2020-02-11 08:49:27.999559028  22.712795  POINT Z (-0.13414 51.52471 114.85894)

The Receiverpoints object is a GeoPandas DataFrame, so can use all its normal methods for geospatial data handling. If you are new to GeoPandas, we suggest `familarising yourself <https://geopandas.org/getting_started/introduction.html>`_ with its core concepts.

Amending Receiverpoint positions
--------------------------------
The receiver positions are those generated from the GNSS receiver, but you may want to input your own depending on their accuracy. The GNSS mapping algorithm is particularly sensitive to the receiver's recorded altitude, which can often be inaccurate due to various reasons relating to satellite geometry.

As an example we are going to double check our elevation estimates by using a `Digital Terrain Model <https://data.gov.uk/dataset/3fc40781-7980-42fc-83d9-0498785c600c/lidar-composite-dtm-2019-1m>`_ (DTM) produced by the UK's Environment Agency and available under an Open Government Licence. This gives ground level heights for the majority of the UK. This requires some geospatial processing of raster data, but it isn't a core part of the tutorial, so you can skip to the next section if it's not of immediate interest.

First, the Receiverpoints must be reprojected to match the coordinate reference system of the digital terrain model (British National Grid with Ordnance Datum Nelwyn)::
 
    >>> gm.geo.pyproj.network.set_network_enabled(True)
    # Ensures the most accurate CRS transform is available, otherwise WGS84 doesn't always transform Z coordinate to BNG correctly
    >>> log_BNG = gm.geo.to_crs(log,7405)
    >>> log_BNG.crs
    <Compound CRS: EPSG:7405>
    Name: OSGB 1936 / British National Grid + ODN height
    Axis Info [cartesian|vertical]:
    - E[east]: Easting (metre)
    - N[north]: Northing (metre)
    - H[up]: Gravity-related height (metre)
    Area of Use:
    - name: United Kingdom (UK) - Great Britain onshore - England and Wales - mainland; Scotland - mainland and Inner Hebrides.
    - bounds: (-7.06, 49.93, 1.8, 58.71)
    Datum: OSGB 1936
    - Ellipsoid: Airy 1830
    - Prime Meridian: Greenwich
    Sub CRS:
    - OSGB 1936 / British National Grid
    - ODN height

Now lookup ground height values using the DTM and substitute back into the ReceiverPoints dataframe::

    >>> import rasterio
    >>> dtm = rasterio.open("./examplefiles/LIDAR-DTM-1m-2019-TQ28se/TQ28se_DTM_1m.tif")
    >>> height = dtm.read(1)
    # get the raster row and colum numbers corresponding to the point locations
    >>> row,col = dtm.index(log_BNG.geometry.x,log_BNG.geometry.y)
    #lookup terrain height for each point and add 1 metre to account for phone held at waist height.
    >>> new_z = height[row, col] + 1 
    >>> import geopandas as gpd
    >>> log_BNG.geometry = gpd.points_from_xy(log_BNG.geometry.x,log_BNG.geometry.y,new_z+1)
    >>> log_BNG[['svid','time','Cn0DbHz','geometry']].head()
      svid                          time    Cn0DbHz                                geometry
    0  G02 2020-02-11 08:49:27.999559028   22.34062  POINT Z (529537.953 182293.216 29.080)
    1  G05 2020-02-11 08:49:27.999559028  26.320181  POINT Z (529537.953 182293.216 29.080)
    2  G07 2020-02-11 08:49:27.999559028  47.322662  POINT Z (529537.953 182293.216 29.080)
    3  G09 2020-02-11 08:49:27.999559028  35.282738  POINT Z (529537.953 182293.216 29.080)
    4  G13 2020-02-11 08:49:27.999559028  22.712795  POINT Z (529537.953 182293.216 29.080)

Once we are happy with our Receiverpoint dataset, it's time to move on to processing it further.

Generating Observations
-----------------------
The next step is to add information about satellite positions, including satellites where the signal wasn't received but which should have been visible.
For the rest of the tutorial, we'll use a `receiverpoint file <https://github.com/Indicative-Data-Science/gnssmapper/blob/master/examplefiles/pilot_study.geojson.zip>`_ created as part of a pilot study::

    >>> import geopandas as gpd
    >>> pilot_log = gpd.read_file("zip://./examplefiles/pilot_study.geojson.zip", driver="GeoJSON")
    >>> pilot_log.time = pilot_log.time.astype('datetime64')
    >>> pilot_log.svid = pilot_log.svid.astype('string')
    # Correcting the altitudes
    >>> pilot_log.geometry=gpd.points_from_xy(pilot_log.geometry.x,pilot_log.geometry.y,80)

In order to retrieve information on the historic positions of satellites, GnssMapper downloads data from the ESA. Downloading and parsing the data is slow, so a local cache is generated, and loaded into memory as required::

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

These are now a set of Observations which can be used in the mapping algorithm. Again this a GeoPandas dataframe, and is quite similar to the Receiverpoints but the geometry has changed from points (representing the receiver position) to a series of rays between the receiver and satellite. Rays are straight lines which represents a direct path from the receiver towards the relevant satellite. They are truncated at 1km in length, in order to minimise inaccuracy upon transformation to a projected CRS (a straight line in a geographic CRS is not a straightline in a projected CRS). 

There are also many more observations, corresponding to the unobserved satellites, which are recorded with a missing signal strength (``Cn0DbHz`` is NaN).

Having processed the data we can save it for analysis. It can be read using GeoPandas, with some minimal processing to ensure datatypes have been read correctly::

    >>> obs.to_file('./examplefiles/obs.geojson', driver="GeoJSON")
    >>> test = gpd.read_file('./examplefiles/obs.geojson')
    >>> test.time = test.time.astype('datetime64')
    >>> test.svid = test.svid.astype('string')

Applying the Mapping Algorithm
------------------------------
The expected map form is another GeoPandas DataFrame, with the geometry now being a collection of 2D polygons, along with a corresponding height column. This represents a simple LOD1 3D map. It can be initialised from a 2D map with a blank height column. For the pilot study, the 2D map was `downloaded <https://api.os.uk/downloads/v1/products/OpenMapLocal/downloads?area=TQ&format=ESRI®+Shapefile&redirect.>`_ from Ordnance Survey's `Open Map Local <https://osdatahub.os.uk/downloads/open/OpenMapLocal>`_  and a building of interest was picked out::

    >>> mymap = gpd.read_file('./examplefiles/OS OpenMap Local (ESRI Shape File) TQ/data/TQ_Building.shp', rows=slice(398502, 398503))
    # We have to add a height column and ensure the polygons are only two dimensional. 
    >>> mymap=gm.geo.drop_z(mymap)
    >>> mymap['height'] = 0
    # The original map CRS is BNG without a vertical datum, we add one so the CRS transform is vaid
    >>> mymap = mymap.set_crs(7405,allow_override=True)
    >>> mymap
                                         ID  FEATCODE                                           geometry  height
    0  000BEF1D-8DAD-4FA5-8EE9-0740DF8C2908     15014  POLYGON ((529673.640 182194.510, 529665.210 18...       0
    

Given a map of floorplates and a set of observations, the height of map elements can be predicted from the observations. GnssMapper implements a bootstrapped four-parameter logistic regression. This fits a four-parameter logistic regression to the data and estimates the height based on model parameters. ::

    >>> gm.predict(mymap,obs)
       lower_bound  mid_point  upper_bound
    0    47.443124  52.645458    57.847791

These are all absolute heights rather than relative to ground level, which is around 30 metres in this case - so the building is in the order of 20 metres high. 
How does this compare to the ground truth? Ordnance Survey data suggests that the absolute height of the building is 55m at the very highest point, and that other parts of the roof are at 47m, so this seems like the algorith has worked relatively well.

How does the algorithm work?
----------------------------
At a very high level, the algorithm uses the Cn0DbHz feature to classify signals as LOS/NLOS. If there is a building blocking the ray, the signal will be missing or weaker (as it is actually received after being reflected off antoher building). The intersection height is also a predictor for LOS - if it is above the actual height of the building the signal should be LOS. It's actually more complicated than this because Cn0DbHz is not particularly accurate at classifying signals, and the interesection height is also inaccurate due to reliability of the receiver position. Nevertheless, it's possible to fit a type of logistic regression for signal height against signal class, and the model parameters relate to the building's height.

We can explore the algorithm further. First we prepare the dataset of intersection heights and fit the models::

    >>> data = gm.algo.prepare_data(mymap, obs)
    >>> data.head()
            0  Cn0DbHz
    0  46.157860      NaN
    1  63.463573      NaN
    2  64.654790      NaN
    3  35.518055      NaN
    4  36.434540      NaN
    >>> learnt_parameters = gm.algo.fit(data[0], data['Cn0DbHz'])
    # These are a timeseries of evolving parameters from two link height and signal strnegth classifiers. We are interested in their final states.
    >>> signalstrength_parameters = learnt_parameters [-1, 0]
    >>> height_parameters = learnt_parameters[-1, 1]

Next we see how the proportion of LOS signals varies with height::

    >>> import pandas as pd
    >>> bins = pd.cut(data[0], bins=range(30, 81))
    >>> from scipy.special import expit, logit
    >>> def inv(param, z):
    ...   return param[2] + logit((z-param[3])/(param[0] - param[3]))/param[1]
    >>> pred = data['Cn0DbHz']>inv(signalstrength_parameters,0.5)
    >>> proportion = pred.groupby(bins).sum() / pred.groupby(bins).count()

Finally we plot this along with the fitted 4-parameter logistic regression::

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(30.5, 80.5, 50)
    >>> def f(param,z):
    ...    return param[3] + (param[0] - param[3]) * expit(param[1] * (z - param[2]))
    >>> z = f(height_parameters,x)
    >>> ax.plot(x, proportion, 'o', color='tab:brown')
    >>> ax.plot(x, z)
    >>> ax.set_xlabel('intersection height (m)')
    >>> ax.set_ylabel('Proportion LOS')
    >>> fig.suptitle('logistic regression on signal classification')
    >>> plt.show()

.. image:: /_static/fit.png
    :width: 500
    :alt: 'Graph of model fit'

The predicted height relates to the position and steepness of the slope of the graph.