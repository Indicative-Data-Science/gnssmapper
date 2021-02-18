import json
import pandas as pd
import numpy as np

from algorithms.FPL import FourParamLogisticRegression
from itertools import islice, cycle,product,accumulate

""" 
=========================================
Mapping Algorithm for building height classification
=========================================

"""
class MapAlgorithm:
    def __init__(self, map, data_filepath=None, observations=None):
        self.map = map
        self.fpath = data_filepath
        self.params={}
        if not observations is None:
            self.data=generate_data(map,observations) # needs to be chnaged to dataframe
            if not self.fpath is None: 
                save_data(self.fpath,self.data)
        else:
            self.data=load_data(self.fpath)

    def update_data(self,building = None):
        #requires observations. Recalculates the data for a specific, for use if the floor plate changes...
        pass

    def reconstruct(self,id_,online=True,**kwargs):
        data = self.visible(id_)
        params = fit(data,self.params.get(id_,starting_params(data)),online,**kwargs)
        self.params[id_]=[list(params[-1,0,:]),list(params[-1,1,:])]    
        return params

    def visible(self, id_):
        """Returns strength and intersection heights of visible signals (currently all) for a building   
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

def load_data(fpath):
    try:
        return pd.read_json(fpath,orient='split',dtype=np.float64)
    except IOError:
        return pd.DataFrame()

def save_data(fpath,data) ->None:
    data.to_json(fpath,orient='split',index=False,indent=4)


def generate_data(map,observations):
    data = map.projected_height(observations)
    data.insert(0,'ss',observations['ss'].to_numpy())
    return data


def fit(data,starting_params_,online,meso_batch: int=50000,iterations: int=4,height_lr=1e-4*np.array([0.1,100,1000,0.1]),SS_lr=1e-4*np.array([2,100,1000,2]),SGD_batch=100):
    """ Applies the reconstruction algorithm

    Parameters
    ----------
    data: [n,2] np array of ss, height observations
    starting_params_ : [2,4] initialisation parameters for the SS and height 4pl's
    meso_batch_: number of mini-batches to run, in online case. links directly to learning rate. For offline case |meso_batch * SGD_batch data| data points used.
    iterations: number of repeats of the bootstrapping process
 
    Returns
    -------
    location : [iterations,2,4] timeseries of parameter estimates.

    """
    #lr,sgd_batch only change online fitting
    SSLR = FourParamLogisticRegression(lr=SS_lr,batch_size=SGD_batch) 
    HeightLR = FourParamLogisticRegression(lr=height_lr,batch_size=SGD_batch)
    (SSLR.a, SSLR.b, SSLR.c, SSLR.d), (HeightLR.a, HeightLR.b, HeightLR.c, HeightLR.d) = starting_params_

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
        y = SSLR.predict(ss)
        param_stream= HeightLR.fit_online(heights, y) if online else HeightLR.fit_offline(heights, y) 
        h_param.extend(param_stream)
        y = heights>HeightLR.c #includes np.nan
        param_stream = SSLR.fit_online(ss, y) if online else SSLR.fit_offline(ss, y)
        ss_param.extend(param_stream)    
        # ss = np.fromiter(islice(ss_,batch),float,count=batch)
        # heights = np.fromiter(islice(heights_,batch),float,count=batch)
        # missing= np.fromiter(islice(missing_,batch),bool,count=batch)
        # ss_intersect= ss[~missing]
        # heights_intersect=heights[~missing]
        # y = SSLR.predict(ss_intersect)
        # param_stream= HeightLR.fit_online(heights_intersect, y)
        # # h_param.append(fill_array(missing,param_stream))
        # h_param.append(param_stream)
        # y = heights_intersect>HeightLR.c
        # param_stream = SSLR.fit_online(ss_intersect, y)
        # # ss_param.append(fill_array(missing,param_stream))
        # ss_param.append(param_stream)
    params = np.stack((ss_param,h_param),axis=1)
        
    return params
    


# def fill_array(mask,values):
#     #extends an array of partial observations
#     x =np.empty((mask.shape[0],4),dtype=float)
#     first_ob = np.min(np.arange(mask.shape[0])[~mask])
#     idx=np.where(~mask,np.arange(mask.shape[0]),first_ob)
#     np.maximum.accumulate(idx,out=idx)
#     x[~mask,:] = values
#     x[mask,:]=x[idx[mask],:]
#     return x

# def fit_offline(data,starting_params_,batch=int(4e6),iterations=1.2e7,convergence_limit=[0.01,0.01,0.01,0.01]):
#     """ Applies the offline reconstruction algorithm

#     Parameters
#     ----------
#     data: [n,2] np array of ss, height observations
    
#     convergence_limit = list of a,b,c,d 4PL tolerances
 
#     Returns
#     -------
#     location : [,2 ,4] timeseries of parameter estimates.

#     """

#     SSLR = FourParamLogisticRegression()
#     HeightLR = FourParamLogisticRegression()
#     (SSLR.a, SSLR.b, SSLR.c, SSLR.d), (HeightLR.a, HeightLR.b, HeightLR.c, HeightLR.d) = starting_params_
    
#     ss = data[~np.isnan(data[:,1]), 0]
#     heights = data[~np.isnan(data[:,1]), 1]
#     param=[starting_params_]
#     converged = False
#     count=0
    
#     while (not converged and count<10):
#         y = SSLR.predict(ss)
#         HeightLR.fit_offline(heights, y)
#         y =  heights > HeightLR.c
#         SSLR.fit_offline(ss, y)
#         param.append([[SSLR.a,SSLR.b,SSLR.c,SSLR.d],[HeightLR.a,HeightLR.b,HeightLR.c,HeightLR.d]])
#         count+=1
#         converged =check_convergence(param,convergence_limit)
#     return np.array(param) 

# def check_convergence(param,limits=[0.01,0.01,0.01,0.01]):
#     current = param[-1][1]
#     prior = param[-2][1]
#     return all([abs(x-y)<l for x,y,l in zip(current,prior,limits)])

def mse(true:float,params):
    lb=params[:,1,2]
    ub=lb +3/ params[:,1,1]
    return ((lb-true)**2 +(ub-true)**2)**0.5

def get_batch_indices(data,batch_size,iterations):
    """returns end indices for batchs with a given amount of non-missing data"""
    indices = (i for i,x in enumerate(data) if x==x) #x==x false if nan 
    return list(islice(indices,0,batch_size*iterations,batch_size))



