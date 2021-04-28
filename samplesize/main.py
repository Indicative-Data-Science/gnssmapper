import gnssmapper as gm
import geopandas as gpd
import pandas as pd
import shapely

import numpy as np
import matplotlib.pyplot as plt

def indicator(data,height,distance=np.inf):
    if distance==np.inf:
        return ~np.isnan(data)
    return ((height-distance<data) & (data<height+distance))


def count(data,height,distance=np.inf):
    """Size metric of dataset for given building height. 
    Minimum number of data points either above or below height, within a distance.
    """
    if distance==np.inf: 
        lower = np.sum(data < height)
        upper = np.sum(data > height)  
    
    else:
        lower = np.sum((data > height - distance) & (data < height))
        upper = np.sum((data > height) & (data < height + distance))
    
    return min (lower,upper)

def window(data,height,count):
    """Size metric of dataset for given building height. 
    Minimum window to ensure count number of datapoints above or below height.
    Counterpart to count metric.
    """       
    upper = np.sort(data[data>height])
    lower = np.sort(data[data<height])[::-1]
    if min(len(upper),len(lower))<count:
        return np.inf
    d_upper = upper[count-1] - height
    d_lower = height - lower[count-1]
    return max (d_upper,d_lower)    


map_= gpd.GeoDataFrame({'height':[10],'geometry':[shapely.wkt.loads("POLYGON((528010 183010, 528010 183000,528000 183000, 528000 183010,528010 183010))")]},crs=7405)
gm.common.check.check_type(map_,'map')
box = shapely.geometry.box(527970, 182970, 528030,183030)
start = pd.Timestamp('2021-01-01 00:00:00')
end = pd.Timestamp('2021-01-01 23:59:59')

print('creating points')
points = gm.sim.point_process(map_,box.bounds,start, end,2000)
gm.geo.pyproj.network.set_network_enabled()

print('creating obs')
obs = gm.observe(points, ['G', 'R', 'C', 'E'])
print('crs update')
obs = gm.geo.to_crs(obs, map_.crs)
obs.time=obs.time.astype('datetime64[s]')
obs['Cn0DbHz']=0
print('getting heights')
heights=gm.algo.prepare_data(map_,obs)
print('merging data')
xy = [l.coords[0][0:2] for l in obs.geometry] 
x,y = tuple(list(x) for x in zip(*xy)) #x-y coords
obs['x']=x
obs['y']=y
obs['z']=heights[0]
obs['d']=((obs.x-528005)**2+(obs.y-183005)**2)**0.5
obs.to_file('samplesize/obs.geojson', driver='GeoJSON')


# obs['i']=indicator(obs['z'],30,5)

# obs['x_floor']=np.floor((obs.x-527970)/5).astype('int8')
# obs['y_floor']=np.floor((obs.y-182970)/5).astype('int8')
# intensity_data = obs.groupby(['x_floor','y_floor']).i.mean().reset_index()
# intensity=np.zeros((12,12))
# intensity[intensity_data.x_floor,intensity_data.y_floor]=intensity_data.i
# # plt.imshow(intensity)

# # plt.show()

# obs['good']=((5<obs.z) & (obs.z<15))*1
# obs[['x','y','z','d','good']].describe()
# d = np.array([x for x,y in zip(obs.d,obs.good) if y])


# x=np.linspace(0,150,num=151)
# y =np.array ([window(obs.z[obs.d<i],10,10) for i in x])
# point_distribution = np.array([sum(obs.d<i) for i in x])

# plt.plot(x,y)
# plt.show()

# plt.plot(x,point_distribution)
# plt.show()



