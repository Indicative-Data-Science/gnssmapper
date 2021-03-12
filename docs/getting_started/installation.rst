Installation
============

GnssMapper depends on a number of geospatial open source libraries (GEOS, GDAL, PROJ)
shared across python packages (Fiona, GeoPandas, PyGeos, PyProj, Shapely). 
Difficulties can arise if the libraries are missing or different packages use different versions. 

The guide below summarises how to install the dependencies to avoid installation problems. If this doesn't work for you, the issues are likely to be with the dependencies, and you should refer to their detailed installation instructions (`Fiona <https://fiona.readthedocs.io/en/latest/README.html#installation>`_, `GeoPandas <https://geopandas.org/getting_started/install.html>`_, 
`PyGeos <https://pygeos.readthedocs.io/en/latest/installation.html>`_, `PyProj <https://pyproj4.github.io/pyproj/stable/installation.html>`_, `Shapely <https://shapely.readthedocs.io/en/latest/project.html#installing-shapely>`_)

Installing with pip
-------------------
GnssMapper can be installed using pip, however if the dependencies are automatically downloaded they are likely to cause errors and they must be **correctly installed first**.

::

    pip install gnssmapper

Installing dependencies using pip
---------------------------------

On Unix
^^^^^^^
On unix-like systems this is relatively straight-forward. Shapely and PyGeos both rely on GEOS (a C library) and must be compiled against a common version. 
GEOS can be installed directly using a system package manager (e.g. Homebrew) and the packages can be built from source::
    
    brew install geos
    #path/to replaced by GEOS path e.g. /usr/local/bin. This can be found by running geos-config --prefix from command line
    GEOS_CONFIG=/usr/local/bin/geos-config pip install --no-binary shapely shapely
    GEOS_CONFIG=/usr/local/bin/geos-config pip install --no-binary pygeos pygeos

The other dependencies can then be installed automatically as part of the installation of gnssmapper::
    
    pip install gnssmapper

On Windows
^^^^^^^^^^
There is no easy approach for Windows systems, but per the installation notes of the underlying packages, the following may work:
    *   GEOS and GDAL libraries can be found at the `OSGeo <https://trac.osgeo.org/osgeo4w/>`_ website. 
    *   Set GEOS_INCLUDE_PATH and GEOS_LIBRARY_PATH environment variables to the installed libraries. 
    *   Build the PyGeos and Shapely packages from source::
  
            pip install --no-binary shapely shapely
            pip install --no-binary pygeos pygeos

    *   A fiona binary is not available from PyPi but can be found `here <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_. This requires the GDAL library mentioned above.
    *   The directory containing the GDAL DLL (gdal304.dll or similar) needs to be in your Windows PATH (e.g. C:\\gdal\\bin). 
    *   The gdal-data directory needs to be in your Windows PATH or the environment variable GDAL_DATA must be set (e.g. C:\\gdal\\bin\\gdal-data).
    *   The environment variable PROJ_LIB must be set to the proj library directory (e.g. C:\\gdal\\bin\\proj6\\share)
    *   The other dependencies can then be installed automatically as part of the installation of gnssmapper::
  
            pip install gnssmapper

Installing dependencies using conda
-----------------------------------
The conda package manager can be used to obtain both the python packages and C libraries::

    conda install --channel conda-forge geopandas
    conda install --channel conda-forge pygeos 
    conda install scipy 

pip can then be used to install gnssmapper, ideally in a `conda environment <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_.

Installing from source
----------------------
A development version of the code is available from the `Github repository <https://github.com/indicative-data-science/gnssmapper>`_.