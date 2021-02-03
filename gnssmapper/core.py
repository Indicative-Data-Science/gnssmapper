""" 
Contains the core of gnssmapper.
A private module. All functions accessible from main namespace

"""

import geopandas as gpd
import gpstime
import pygeos


def check_valid_observations(obs: gpd.GeoDataFrame) -> bool:
    """Checks a geodataframe is a valid set of observations."""
    pass


def check_valid_receiverpoints(points: gpd.GeoDataFrame) -> None:
    """Checks a geodataframe is a valid set of receiver points."""
    # fatal errors
    if points.is_empty.any():
        raise ValueError('Missing receiver locations')

    if not points.geom_type.eq("Point").all():
        raise ValueError(
            'Invalid receiver locations (expecting point geometries)')

    if points.z.is_na().any():
        raise ValueError('Missing z coordinate in receiver locations')

    if 'time' not in points.columns:
        raise ValueError('"time" column missing')
    if points.['time'].dtype != "datetime":
        raise ValueError('datatype of times column is not datetime')

    # warnings
    # if crs is 2d and will be promoted....
    if 'svid' in points.columns:
        check_constellations(points['svid'],constants.supported_constellations)

    return None


def check_constellations(svid: pd.Series,expected: set[str]) -> None:
    unsupported = set(svid.str[0].unique()) - expected
    if ~unsupported:
        warnings.warn(f'Includes unsupported constellations: {unsupported}\n')
    return None


def observe(points: gpd.GeoDataFrame, constellations: set[str] = []) -> gpd.GeoDataFrame:
    """Generates a set of observations from a receiverpoints dataframe.

    Observations includes all above horizon svids, not only those measured in receiverpoints dataframe.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        gnss receiverpoints including:
            receiver position (as point geometry)
            time (utc format)

    constellations : set[str], optional
        constellations supported by gnss receiver. If not supplied it is inferred from the measured receiverpoints.

    Returns
    -------
    gpd.GeoDataFrame
        observations including:
        geometry (linestring from receiver in direction of satellite)
        time
        sv
        signal features
    """
    check_valid_receiverpoints(points)
    check_constellations(constellations,constants.supported_constellations)
    measured_constellations = set(points['svid'].str[0].unique())

    if ~constellations:
        if ~measured_constellations:
            raise ValueError(
                "Supported constellations cannot be inferred from receiverpoints and must be supplied")
        else:
            constellations = measured_constellations

    gps_time = gpstime.utc_to_gps(points['time'])
    sd = SatelliteData()
    #Generate dataframe of all svids supported by receiver
    svids = sd.name_satellites(gps_time).explode().dropna().reset_index(name='gps_time')  
    svids = svids[svids['svid'].str[0].isin(constellations)]

    #locate the satellites
    sats = sd.locate_satellites(svids['svid'], svids['gps_time'])

    #revert to utc time
    sats['time'] = gpstime.gps_to_utc(obs['gps_time'])
    sats=sats.set_index(['time','svid'])
    
    #convert points into geocentric WGS and merge
    receiver_df = points.to_crs('EPSG:4978').set_index(['time','svid'])
    obs = receiver_df.merge(sats)
    # # rec_df = pd.DataFrame(points.drop('geometry',axis=1))
    # # rec_df.assign(x=receiver_location.x,y=receiver_location.y,z=receiver_location.z)
    # # rec_df=rec_df

    # #join observations to receiver points
    # obs = obs.join(rec_df)
    #create new geometry
    rays= rays(obs.x,obs.y,obs.z,obs.sv_x,obs.sv_y,obs.sv_z)
    obs.drop(columns=['x','y','z','sv_x','sv_y','sv_z'])
    obs = gpd.GeoDataFrame(obs,crs='EPSG:4978',geometry = rays)

    # filter observations
    obs = filter_elevation(obs,constants.minimum_elevation)

    check_valid_observations(obs)
    return obs

def rays(x,y,z,sv_x,sv_y,sv_z):
        

        return pygeos.creation.linestrings(x,y,z)


        receiver=observations.loc[:,["x","y","z"]].to_numpy().tolist()
        sat=observations.loc[:,["sv_x","sv_y","sv_z"]].to_numpy().tolist()
        return  (shapely.geometry.LineString([r,s]) for r,s in zip(receiver,sat))
