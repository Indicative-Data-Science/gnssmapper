import gnssmapper as gm
import geopandas as gpd
from shapely.geometry import Polygon
import cProfile

# p = cProfile.run('gm.read_gnsslogger("./examplefiles/gnss_log_2020_02_11_08_49_29.txt")')
# p.sort_stats('cumulative').print_stats(20)
log = gm.read_gnsslogger("./examplefiles/gnss_log_2020_02_11_08_49_29.txt")
missing = log[2:10]
obs = gm.observe(missing)
missing['svid']

# map_ = gpd.read_file('./examplefiles/mastermap-topo_3473984_0.gml')
# map_.geometry = [Polygon(a) for a in map_.geometry]
# map_['height'] = 50
# gm.common.check.map(map_)

# start = gpd.pd.Timestamp('2020-02-11T11')
# end = gpd.pd.Timestamp('2020-02-11T12')
# # sim = gm.simulate(map_, "point_process", 100, start, end, method_args={'receiver_offset': 2})
# rp = gm.sim.point_process(map_, 100, start, end, **{'receiver_offset': 2})
# sim_obs = gm.observe(rp, set(['C', 'R', 'E', 'G']))


# sim_obs2 = gm.geo.to_crs(sim_obs, map_.crs)
# gm.geo.to_crs(sim_obs.loc[10398:10398,:],map_.crs)
# sim_obs.loc[10398:10398,:].to_crs(map_.crs)
# map_.geometry.to_crs(gm.common.constants.epsg_wgs84)
# map_.geometry.to_crs(gm.common.constants.epsg_wgs84_cart)
# gm.geo.to_crs(map_.geometry,gm.common.constants.epsg_wgs84)  #doesn't work - this is reversed!!
# gm.geo.to_crs(map_.geometry,gm.common.constants.epsg_wgs84_cart) 

# test = sim_obs.geometry[4982]
# test
# test[4981]
# sim_obs2.loc[4980:4982,:]

# sim_obs
# interim=gm.geo.to_crs(sim_obs,gm.common.constants.epsg_wgs84)
# interim
# interim.to_crs(map_.crs)





# import pyproj
# wgs=pyproj.crs.CRS(gm.common.constants.epsg_wgs84)
# cart=pyproj.crs.CRS(gm.common.constants.epsg_wgs84_cart)

# sim_obs.to_crs(map_.crs)
# sim_obs2
# rp.crs
# list(sim_obs.at[10398, 'geometry'].coords)

# gm.geo.to_crs((rp.loc[99,'geometry'].coords), gm.common.constants.epsg_wgs84_cart)

# mapwgs = map_.to_crs(gm.common.constants.epsg_wgs84)


# test = gm.geo.to_crs(rp, gm.common.constants.epsg_wgs84)
# back = gm.geo.to_crs(test, rp.crs)
# back