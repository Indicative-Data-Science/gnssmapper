import json
import pandas as pd
import numpy as np

from algorithms.FPL import FourParamLogisticRegression
from itertools import islice, cycle

""" 
=========================================
Mapping Algorithm for building height classification
=========================================

"""
class MapAlgorithm:
    def __init__(self, map, data_filepath, observations=None):
        self.map = map
        self.fpath = data_filepath
        self.params={}
        if not observations is None:
            self.data=generate_data(map,observations) # needs to be chnaged to dataframe
            save_data(self.fpath,self.data)
        else:
            self.data=load_data(self.fpath)

    def update_data(self,building = None):
        #requires observations. Recalculates the data for a specific, for use if the floor plate changes...
        pass

    def reconstruct(self,id_,online=True):
        data = self.visible(id_)
        if online:
            params = fit_online(data,self.params.get(id_,starting_params(data)))
        else:
            params = fit_offline(data,self.params.get(id_,starting_params(data)))
    
        self.params[id_]=[list(params[0,-1,:]),list(params[1,-1,:])]    
        return params

    def visible(self, id_):
        """Returns the data of visible signals (currently all) for a building   
        """
        visible = self.data.loc[:,id_] #don't want to get rid of missing because it affects speed metrics
        return np.column_stack((self.data['ss'],visible))


    @property
    def heights(self):
        def height(id_):
            _,param = self.params.get(id_,[[np.nan]*4,[np.nan]*4])
            lb= param[2]
            ub = lb + 3/param[1]
            return [lb,ub]
        return np.array([height(i) for i in self.map.buildingID])
    
def starting_params(data):
    """generates parameters with balanced classes"""
    ss,h = np.median(data[~np.isnan(data[:,0])],axis=0)
    b = 0.1
    return [[0.8,b,ss,0.2],[0.8,b,h,0.2]]

def load_data(fpath):
    try:
        with open(fpath) as json_file:
            return pd.read_json(json_file,orient='split')
    except IOError:
        return pd.DataFrame()

def save_data(fpath,data) ->None:
        with open(fpath) as outfile:
            data.to_json(outfile,orient='split',index=False,indent=4)


def generate_data(map,observations):
    data = map.projected_height(observations)
    data.insert(0,'ss',observations['ss'].to_numpy())
    return data


def fit_online(data,starting_params_,batch=int(4e6),iterations=1.2e7,height_lr=1e-4*np.array([0.1,100,1000,0.1]),SS_lr=1e-4*np.array([2,100,1000,2]),SGD_batch=100):
    """ Applies the reconstruction algorithm

    Parameters
    ----------
    data: [n,2] np array of ss, height observations
    
 
    Returns
    -------
    location : [,2,4] timeseries of parameter estimates.

    """
    
    SSLR = FourParamLogisticRegression(lr=SS_lr,batch_size=SGD_batch)
    HeightLR = FourParamLogisticRegression(lr=height_lr,batch_size=SGD_batch)
    (SSLR.a, SSLR.b, SSLR.c, SSLR.d), (HeightLR.a, HeightLR.b, HeightLR.c, HeightLR.d) = starting_params_

    ss_ = cycle(data[:, 0])
    heights_ = cycle(data[:, 1])
    missing_ = cycle(np.isnan(data[:, 1]))
    h_param = [starting_params_[1]]
    ss_param = [starting_params_[0]]

    for _ in np.arange(0,iterations,batch):
        ss = np.fromiter(islice(ss_,batch),float,count=batch)
        heights = np.fromiter(islice(heights_,batch),float,count=batch)
        missing= np.fromiter(islice(missing_,batch),bool,count=batch)
        ss_intersect= ss[~missing]
        heights_intersect=heights[~missing]
        y = SSLR.predict(ss_intersect)
        param_stream= HeightLR.fit_online(heights_intersect, y)
        h_param.append(fill_array(missing,param_stream))
        y = heights_intersect>HeightLR.c
        param_stream = SSLR.fit_online(ss_intersect, y)
        ss_param.append(fill_array(missing,param_stream))
    
    params = np.stack([np.vstack(tuple(ss_param)),np.vstack(tuple(h_param))])
        
    return np.swapaxes(params,0,1)

def fill_array(mask,values):
    #extends an array of partial observations
    x =np.empty_like(mask)
    idx=np.where(~mask,np.arange(mask.shape[0]),0)
    np.maximum.accumulate(idx,out=idx)
    x[~mask] = values
    x[mask]=x[idx[mask]]
    return x

def fit_offline(data,starting_params_,convergence_limit=[0.01,0.01,0.01,0.01]):
    """ Applies the offline reconstruction algorithm

    Parameters
    ----------
    data: [n,2] np array of ss, height observations
    
    convergence_limit = list of a,b,c,d 4PL tolerances
 
    Returns
    -------
    location : [,2 ,4] timeseries of parameter estimates.

    """

    SSLR = FourParamLogisticRegression()
    HeightLR = FourParamLogisticRegression()
    (SSLR.a, SSLR.b, SSLR.c, SSLR.d), (HeightLR.a, HeightLR.b, HeightLR.c, HeightLR.d) = starting_params_
    
    ss = data[~np.isnan(data[:,1]), 0]
    heights = data[~np.isnan(data[:,1]), 1]
    param=[starting_params_]
    converged = False
    count=0
    
    while (not converged and count<10):
        y = SSLR.predict(ss)
        HeightLR.fit_offline(heights, y)
        y =  heights > HeightLR.c
        SSLR.fit_offline(ss, y)
        param.append([[SSLR.a,SSLR.b,SSLR.c,SSLR.d],[HeightLR.a,HeightLR.b,HeightLR.c,HeightLR.d]])
        count+=1
        converged =check_convergence(param,convergence_limit)
    return np.array(param) 

def check_convergence(param,limits):
    current = param[-1,1,:]
    prior = param[-2,1,:]
    return all([abs(x-y)<l for x,y,l in zip(current,prior,limits)])