""" 
Mapping algorithm methods

"""

from itertools import islice, cycle, product, accumulate
import json

import geopandas as gpd
import numpy as np
import pandas as pd


from gnssmapper.algo.FPL import FourParamLogisticRegression as fpl
import gnssmapper.common as cm
from gnssmapper.geo import projected_height



def predict(map_: gpd.GeoDataFrame, obs: gpd.GeoDataFrame,**kwargs) -> pd.DataFrame:
    """Predicts heights for each map object based on a set of observations. 


    Parameters
    ----------
    map_ : gpd.GeoDataFrame
        map with heights to be estimated
    obs : gpd.GeoDataFrame
        observations use to fit algorithm.

    Returns
    -------
    pd.DataFrame
        height of each map object
    """    
    cm.check.map(map_)
    cm.check.observations(obs)

    data = prepare_data(map_, obs)
    return fit_data(data, **kwargs)

def prepare_data(map_: gpd.GeoDataFrame, obs: gpd.GeoDataFrame) -> pd.DataFrame:
    """ Returns a dataframe of intersection heights and signal features.
    
    Row indexed using observation index, whereas columns refer to map index."""    

    data = projected_height(map_,obs.geometry)
    data['Cn0DbHz']=obs['Cn0DbHz']
    return data

def fit_data(data:pd.DataFrame, **kwargs)->pd.DataFrame:
    index = [c for c in data.columns if c!='Cn0DbHz']
    params = [fit(height=data[i], ss=data['Cn0DbHz'], **kwargs)[-1,:,:] for i in index]
    heights = np.array([_heights(p) for p in params])
    return pd.DataFrame(heights,index=index,columns=['lower_bound','mid_point','upper_bound'])

def _heights(param):
    """ Converts model parameters into height bounds """
    _, height_param = param
    lb= height_param[2]
    ub = lb + 3/height_param[1]
    return (lb, (lb + ub) / 2, ub)

def fit(height: np.array, ss: np.array, iterations: int = 4, batch_size: int = 0, online: bool = False, 
        starting_params: list = [], online_params: dict = {
            'ss_lr': [0.0002, .01, .1, .0002],'height_lr': [0.00001, .01, .1, .00001], 'batch_size': 100,
            }) -> np.array:
    """Applies the reconstruction algorithm to estimate a building's height.

    Parameters
    ----------
    height : np.array
        intersection height of observations with the building
    ss : np.array
        signal strength 
    iterations: int,optional
        number of times to repeat the bootstrapping process, by default 4
    batch_size: int,optional
        Number of datapoints to use in an iteration of the algorithm, by default 0 uses the entire dataset.
    online : bool, optional
        use online fitting, else offline
    starting_params: list, optional
        starting parameters [ss_params,height_params] for the algorithm, otherwise generated automatically. Each param is a tuple (a,b,c,d)
    online_params : dict, optional
        optional arguments for the online fitting method
            ss_lr: signal strength learning rate, 
            height_lr: height learning rate,
            batch_size: batch_size for stochastic gradient descent.

    Returns
    -------
    np.array
        [n,2,4] of parameter estimates:
            0-axis is parameter estimates over data used(batch_size * iterations), 
            1-axis is model (0 - SS, 1 - height),
            2-axis are the 4pl parameters (a,b,c,d).
    """    
    if height.shape[0] != ss.shape[0]:
        raise ValueError('height and ss vectors differ in length')

    #initialise models
    model_args = {'ss_lr': [0.0002, .01, .1, .0002], 'height_lr': [0.00001, .01, .1, .00001], 'batch_size': 100}
    model_args.update(online_params)
    if  not starting_params:
        starting_params = [(0.8,0.1,np.nanmedian(ss),0.2),(0.8,0.1,np.nanmedian(height),0.2)]

    ss_model = fpl(lr=model_args['ss_lr'],batch_size=model_args['batch_size'],initial_param=starting_params[0])
    height_model = fpl(lr=model_args['height_lr'],batch_size=model_args['batch_size'],initial_param=starting_params[1])

    if batch_size == 0:
        batch_size = int(height.shape[0])
    
    total_size = batch_size * iterations

    ss_ = np.fromiter(islice(cycle(ss),total_size),float,count=total_size) 
    height_ = np.fromiter(islice(cycle(height),total_size),float,count=total_size) 
    
    h_param = []
    ss_param = []

    for pos in range(0,total_size,batch_size):
        h = height_[pos:pos+batch_size]
        s = ss_[pos:pos + batch_size]
        #fit the height model
        y = ss_model.predict(s)
        param_stream= height_model.fit_online(h, y) if online else height_model.fit_offline(h, y) 
        h_param.extend(param_stream)
        #fit the ss model
        y = h>height_model.param[2] 
        param_stream = ss_model.fit_online(s, y) if online else ss_model.fit_offline(s, y)
        ss_param.extend(param_stream)    
    params = np.stack((ss_param,h_param),axis=1)
        
    return params





