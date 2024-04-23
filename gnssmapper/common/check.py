""" This module provides checks that gnssmapper objects are valid.

Objects {rays, receiver_points, observations, maps} are special geoDataFrames.
Not implemented as classes because of the difficulty of subclassing Pandas dataframes.
"""

import warnings
from typing import NewType, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj.crs
import shapely

Rays = NewType("Rays", gpd.GeoSeries)
ReceiverPoints = NewType("ReceiverPoints", gpd.GeoDataFrame)
Observations = NewType("Observations", gpd.GeoDataFrame)
Map = NewType("Map", gpd.GeoDataFrame)


def _rays(obj: Rays) -> dict:
    if not isinstance(obj, gpd.GeoSeries):
        return {"Not a GeoSeries": True}
    else:
        return {
            "Missing geometries": obj.is_empty.any(),
            "Expecting Linestrings": not obj.geom_type.eq("LineString").all(),
            "Missing z coordinates": not obj.has_z.all(),
            "More than 2 points in Linestring": np.not_equal(
                shapely.count_coordinates(obj.array.data), 2 * len(obj)
            ).any(),
        }


def _receiverpoints(obj: ReceiverPoints) -> dict:
    if not isinstance(obj, gpd.GeoDataFrame):
        return {"Not a GeoDataFrame": True}
    else:
        return {
            "Missing geometries": obj.geometry.is_empty.any(),
            "Expecting obj": not obj.geom_type.eq("Point").all(),
            "Missing z coordinates": not obj.has_z.all(),
            '"time" column missing or not datetime': (
                ("time" not in obj.columns) or (obj["time"].dtype != "datetime64[ns]")
            ),
        }


def _observations(obj: Observations) -> dict:
    if not isinstance(obj, gpd.GeoDataFrame):
        return {"Not a GeoDataFrame": True}
    else:
        tests = _rays(obj.geometry)
        tests.update(
            {
                '"svid" column missing or not string': ("svid" not in obj.columns)
                or (
                    (obj["svid"].dtype != "object") and (obj["svid"].dtype != "string")
                ),
                '"time" column missing or not datetime': ("time" not in obj.columns)
                or (obj["time"].dtype != "datetime64[ns]"),
            }
        )
        return tests


def _map(obj: Map) -> dict:
    if not isinstance(obj, gpd.GeoDataFrame):
        return {"Not a GeoDataFrame": True}
    else:
        return {
            "Missing geometries": obj.geometry.is_empty.any(),
            "Expecting Polygons": not obj.geom_type.eq("Polygon").all(),
            "Unexpected z coordinates": obj.has_z.any(),
            '"height" column missing or not numeric': ("height" not in obj.columns)
            or (obj["height"].dtype != "float" and obj["height"].dtype != "int"),
        }


_check_dispatcher = {
    "rays": _rays,
    "receiverpoints": _receiverpoints,
    "observations": _observations,
    "map": _map,
}


def check_type(
    obj, object_type: str = None, raise_errors: bool = False
) -> Union[str, bool]:
    if object_type:
        try:
            test_results = _check_dispatcher[object_type](obj)
        except KeyError:
            raise ValueError("Invalid object_type specified")
        if raise_errors:
            _raise(test_results)
        return not any(test_results.values())

    else:
        return _infer_type(obj)


def _infer_type(obj) -> str:
    if not isinstance(obj, gpd.GeoSeries):
        return "rays" if check_type(obj, "rays") else None

    if isinstance(obj, gpd.GeoDataFrame):
        if obj[0,].geomtype == "Point":
            return "receiverpoints" if check_type(obj, "receiverpoints") else None
        if obj[0,].geomtype == "LineString":
            return "observations" if check_type(obj, "observations") else None
        if obj[0,].geomtype == "Polygon":
            return "map_" if check_type(obj, "map_") else None

    return "None"


def _raise(tests: dict) -> None:
    errors = [k for k, v in tests.items() if v]
    if errors:
        text = ", ".join(errors)
        raise AttributeError(text)
    return None


def crs(crs_: pyproj.crs.CRS) -> None:
    """Warns if crs is 2D and will be promoted for transforms"""
    if len(crs_.axis_info) == 2:
        warnings.warn("2D crs used. Will be promoted for transforms.")
    return None


def constellations(svid: Union[pd.Series, set[str]], expected: set[str]) -> None:
    if isinstance(svid, pd.Series):
        present = set(svid.str[0].unique())
    else:
        present = set(svid)

    unsupported = present - expected
    if unsupported:
        warnings.warn(f"Includes unsupported constellations: {unsupported}")
    return None
