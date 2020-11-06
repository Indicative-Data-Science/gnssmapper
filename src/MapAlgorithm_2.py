import numpy as np
from copy import deepcopy
from FourParamLR import FourParamLogisticRegression
from itertools import islice, cycle
import matplotlib.pyplot as plt

""" 
=========================================
Mapping Algorithm for building height classification
=========================================

"""
# helper function to construct dataframe with correct structure
def MapData(building=None,observation=None,ss=None,z=None):
    #change length of empty arrays to non-zero if needed 
    params=[building,observation,ss,z]
    n= max([0]+ [len(p) for p in params if p is not None])
    zero_params=[
        np.zeros( shape=(n, ) ,dtype=np.str),
        np.zeros( shape=(n, ) ,dtype=np.str),
        np.zeros( shape=(n, ) ,dtype=np.float64),
        np.zeros( shape=(n, ) ,dtype=np.float64)
    ]

    params=[p if p is not None else q for p,q in zip(params,zero_params)]
    
    return pd.DataFrame({
            'building':params[0],
            'observation':params[1],
            'ss':params[2],
            'z':params[3]
            })


class MapAlgorithm:
    """This class performs building height classification iteratively in a semi-supervised manner
     using signal strengths and signal intersection heights. """

    def __init__(self, map, observations,data={},params={}):
        self.map = map
        self.buildings = list(map.buildings)
        self.observations = observations
        self.data=data
        self.params=params
    
    def key(self,building):
        return next(str(i) for i, j in enumerate(self.buildings) if j is building)        

    def updata_all(self):
        for building in self.buildings:
            self.update(building)

    def update(self,building):
        buildingID=np.full(shape=(len(self.observations),),fill_value= self.key(building))
        obs = self.observations.apply(lamba x: x["svid"] +str(x["t"]),axis=1)     
        z = self.map.get_height(self.observations,[building]*len(self.observations))
        update=MapData(building=buildingID,observation=obs,ss=self.observations["ss"],z=z)
        unchanged = self.data.loc[self.data["building"]!=buildingID]
        self.data = pd.concat([unchanged,update], axis=0,ignore_index=True)


    def heights(self,buildings=None):
        if buildings is None:
            buildings=self.buildings
        return np.array([self.height(b) for b in buildings])
    
    def height(self,building):
        _,param = self.params.get(self.key(building),[[np.nan]*4,[np.nan]*4])
        lb= param[2]
        ub = lb + 3/param[1]
        return np.array([lb,ub])

    def startingParams(self,building):
        """generates parameters with balanced classes"""
        data=self.data[self.key(building)]
        ss,h = np.median(data[~np.isnan(data[:,0])],axis=0)
        b = 0.1
        return [[0.8,b,ss,0.2],[0.8,b,h,0.2]]

def fill_array(mask,values):
    x =np.empty_like(mask)
    idx=np.where(~mask,np.arange(mask.shape[0]),0)
    np.maximum.accumulate(idx,out=idx)
    x[~mask] = values
    x[mask]=x[idx[mask]]
    return x
