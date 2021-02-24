""" 
Mapping algorithm methods

"""

from itertools import islice, cycle, product, accumulate
import json

import pandas as pd
import numpy as np

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
    index = data.columns
    params = (fit(height=data.i, ss=data['Cn0DbHz'], **kwargs) for i in index)
    heights = 
    return pd.DataFrame(heights,index=index)

def _heights(param):
    """ Converts model parameters into height bounds """
    ss_param, height_param = param
    lb= height_param[2]
        ub = lb + 3/height_param[1]
    return (lb, (lb + ub) / 2, ub)

def fit(height: np.array, ss: np.array, method: str = 'offline', bootstrap_batch_size
        online_params: dict = {
            'ss': {'lr': [0.0002, .01, .1, .0002], 'batch_size': 100)},
            'height': {'lr': [0.00001, .01, .1, .00001], 'batch_size': 100)}
            }) -> np.array:
    """Applies the reconstruction algorithm to estimate a building's height.

    Parameters
    ----------
    height : np.array
        intersection height of observations with the building
    ss : np.array
        signal strength 
    method : str, optional
        which fitting method to use from {offline, online} , by default 'offline'
    online_params : dict, optional
        optional arguments for the online fitting method.

    starting_params_ : [2,4] initialisation parameters for the SS and height 4pl's
    meso_batch_: number of mini-batches to run, in online case. links directly to learning rate. For offline case |meso_batch * SGD_batch data| data points used.
    iterations: number of repeats of the bootstrapping process

    Returns
    -------
    np.array
        either [iterations,2,4] timeseries of parameter estimates.
    """    

    starting_params_, online, meso_batch: int = 50000, iterations: int = 4,
    
    #initialise models
    ss_model = fpl(**online_params['ss'])
    height_model = fpl(**online_params['height'])
    
    batch_size=SGD_batch*meso_batch
    idx = get_batch_indices(cycle(data[:, 1]),batch_size,iterations)
    end = [i+1 for i in idx] #in order to include the final item
    start = [0] + end[:-1]

    ss_ = np.fromiter(islice(cycle(data[:, 0]),end[-1]),float,count=end[-1]) 
    heights_ = np.fromiter(islice(cycle(data[:, 1]),end[-1]),float,count=end[-1])
    
    h_param = []
    ss_param = []

    for s,e in zip(start,end):
        heights = heights_[s:e]
        ss = ss_[s:e]
        y = ss_model.predict(ss)
        param_stream= height_model.fit_online(heights, y) if online else height_model.fit_offline(heights, y) 
        h_param.extend(param_stream)
        y = heights>height_model.c #includes np.nan
        param_stream = ss_model.fit_online(ss, y) if online else ss_model.fit_offline(ss, y)
        ss_param.extend(param_stream)    
    params = np.stack((ss_param,h_param),axis=1)
        
    return params



def _get_batch_indices(data,batch_size,iterations):
    """returns end indices for batchs with a given amount of non-missing data"""
    indices = (i for i,x in enumerate(data) if x==x) #x==x false if nan 
    return list(islice(indices,0,batch_size*iterations,batch_size))

    def reconstruct(self,id_,online=True,**kwargs):
        data = self.visible(id_)
        params = fit(data,self.params.get(id_,starting_params(data)),online,**kwargs)
        self.params[id_]=[list(params[-1,0,:]),list(params[-1,1,:])]    
        return params



    def performance(self,id_,sample_size,repetitions,online=True,**kwargs):
        data = self.visible(id_)
        true_height = self.map.heights[self.map.buildingID.index(id_)]
        
        samples = (np.random.permutation(data)[:sample_size,:] for _ in range(repetitions))
        results = [fit(s,self.params.get(id_,starting_params(s)),online,**kwargs) for s in samples]
        mse_ =  [mse(true_height,r) for r in results]
        average_mse = np.mean(np.stack(mse_),axis=0)
        return average_mse

def starting_params(data):
    """generates parameters with balanced classes"""
    ss,h = np.median(data[~np.isnan(data[:,0])],axis=0)
    # ss=20
    b = 0.1
    return [[0.8,b,ss,0.2],[0.8,b,h,0.2]]






    


def mse(true:float,params):
    lb=params[:,1,2]
    ub=lb +3/ params[:,1,1]
    return ((lb-true)**2 +(ub-true)**2)**0.5





