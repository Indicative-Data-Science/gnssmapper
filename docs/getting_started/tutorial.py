""" Generates info for tutorial """
#Needs updating for change in filepath

import gnssmapper as gm
log = gm.read_gnsslogger("./examplefiles/gnss_log_2020_02_11_08_49_29.txt")
log[['svid','time','Cn0DbHz','geometry']].head()


#Ensure the most accurate CRS transform is available, otherwise WGS84 doesn't transform Z coordinate to BNG correctly
gm.geo.pyproj.network.set_network_enabled(True)
log_BNG = gm.geo.to_crs(log,7405)
log_BNG.crs

#Use the rasterio package to read a raster digital terrain model
import rasterio
dtm = rasterio.open("./examplefiles/LIDAR-DTM-1m-2019-TQ28se/TQ28se_DTM_1m.tif")
height = dtm.read(1)
# get the row and colum numbers correpsonding to the point locations
row,col = dtm.index(log_BNG.geometry.x,log_BNG.geometry.y)
#lookup terrain height for each point and add 1 metre to account for phone held at waist height.
new_z = height[row, col] + 1 
import geopandas as gpd
log_BNG.geometry = gpd.points_from_xy(log_BNG.geometry.x, log_BNG.geometry.y, new_z + 1)
log_BNG[['svid','time','Cn0DbHz','geometry']].head()

#Preparing for next part of tutorial
obs = gpd.read_file('./examplefiles/obs.geojson')
obs.time = obs.time.astype('datetime64')
obs.svid = obs.svid.astype('string')


# pilot_log = gpd.read_file("zip://./examplefiles/pilot_study.geojson.zip", driver="GeoJSON")
# pilot_log.head()
# len(pilot_log)
# len(obs)
# obs.Cn0DbHz.isna().value_counts()

# import pyproj
# import pandas as pd
# import numpy as np


# bbox = (529466.2 , 182191.0,  529655.0,  182355.85)
# map_ = gpd.read_file('./examplefiles/OS OpenMap Local (ESRI Shape File) TQ/data/TQ_Building.shp')

# map_.describe()
# map_.head()
# map_[map_.ID=='000BEF1D-8DAD-4FA5-8EE9-0740DF8C2908']

mymap = gpd.read_file('./examplefiles/OS OpenMap Local (ESRI Shape File) TQ/data/TQ_Building.shp', rows=slice(398502, 398503))
mymap=gm.geo.drop_z(mymap)
mymap['height'] = 0
mymap = mymap.set_crs(7405, allow_override=True)
mymap.crs
gm.geo.pyproj.network.set_network_enabled(True)
gm.predict(mymap,obs)

data = gm.algo.prepare_data(mymap, obs)
data.head()
learnt_parameters = gm.algo.fit(data[0], data['Cn0DbHz'])
signalstrength_parameters = learnt_parameters [-1, 0]
height_parameters = learnt_parameters[-1, 1]

# first we see how the proportion of LOS signals varies with height
import pandas as pd
bins = pd.cut(data[0], bins=range(30, 81))

from scipy.special import expit, logit
def inv(param, z):
   return param[2] + logit((z-param[3])/(param[0] - param[3]))/param[1]


pred = data['Cn0DbHz']>inv(signalstrength_parameters,0.5)
proportion = pred.groupby(bins).sum() / pred.groupby(bins).count()

#now we plot this and include the fitted 4-parameter logistic regression.
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.linspace(30.5, 80.5, 50)

def f(param,z):
    return param[3] + (param[0] - param[3]) * expit(param[1] * (z - param[2]))

z = f(height_parameters,x)
ax.plot(x, proportion, 'o', color='tab:brown')
ax.plot(x, z)
ax.set_xlabel('intersection height (m)')
ax.set_ylabel('Proportion LOS')
fig.suptitle('logistic regression on signal classification')
plt.show()





# import matplotlib.pyplot as plt
# base = map_.plot(color='white', edgecolor='black')
# obs.plot(ax=base, color='red')
# plt.xlim([529460, 529660])
# plt.ylim([182175,182375])
# plt.show()
import pandas as pd
start = pd.Timestamp('2020-02-11T11')
end = pd.Timestamp('2020-02-11T12')
sim = gm.simulate(map_, "point_process", 100, start, end)
sim.head()


rp = gm.sim.point_process(map_, 100, start, end, **{'receiver_offset': 2})
sim_obs = gm.observe(rp, set(['C', 'R', 'E', 'G']))





