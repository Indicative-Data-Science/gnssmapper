""" This module provides file read and write capabilities for gnssmapper objects.
    Currently unused! - just use the geopandas methods and tweak datatypes on loading...
"""

import geopandas as gpd


def to_file(df, filename, driver="GPKG", schema=None, index=None, **kwargs):
    """Write the ``GeoDataFrame`` to a file.

    By default, a GeoPackage file is written, but any OGR data source
    supported by Fiona can be written. Support for datetime fields varies between drivers.

    Parameters
    ----------
    df: GeoDataFrame to be written.
    filename : string
        File path or file handle to write to.
    driver : string, default: 'GPKG'
        The OGR format driver used to write the vector file.
    schema : dict, default: None
        If specified, the schema dictionary is passed to Fiona to
        better control how the file is written.
    index : bool, default None
        If True, write index into one or more columns (for MultiIndex).
        Default None writes the index into one or more columns only if
        the index is named, is a MultiIndex, or has a non-integer data
        type. If False, no index is written.

    Notes
    -----
    The extra keyword arguments ``**kwargs`` are passed to fiona.open and
    can be used to write to multi-layer data, store data within archives
    (zip files), etc.

    The format drivers will attempt to detect the encoding of your data, but
    may fail. In this case, the proper encoding can be specified explicitly
    by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.
    """
    # make any changes
    data = df.copy()

    # import fiona
    # fiona.Collection.schema(l)
    # infer_schema
    # _geometry_types
    # look like they can help.

    # fiona

    # check if datframe of specific type

    data.to_file()
    final = {}

    final["Cn0DbHz"] = final["Cn0DbHz"].astype("float64")  # otherwise won't write)
    final["time"] = final.time.astype("datetime64[s]")


def read_file(filename, bbox=None, mask=None, rows=None, **kwargs):
    """Returns a GeoDataFrame from a file or URL and checks type against gnssmapper objects.

    Parameters
    ----------
    filename : str, path object or file-like object
        Either the absolute or relative path to the file or URL to
        be opened, or any object with a read() method (such as an open file
        or StringIO)
    bbox : tuple | GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter features by given bounding box, GeoSeries, GeoDataFrame or a
        shapely geometry. CRS mis-matches are resolved if given a GeoSeries
        or GeoDataFrame. Cannot be used with mask.
    mask : dict | GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter for features that intersect with the given dict-like geojson
        geometry, GeoSeries, GeoDataFrame or shapely geometry.
        CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
        Cannot be used with bbox.
    rows : int or slice, default None
        Load in specific rows by passing an integer (first `n` rows) or a
        slice() object.
    **kwargs :
        Keyword args to be passed to the `open` or `BytesCollection` method
        in the fiona library when opening the file. For more information on
        possible keywords, type:
        ``import fiona; help(fiona.open)``
    """

    output = gpd.read_file(filename, bbox=bbox, mask=mask, rows=rows, **kwargs)

    return output
