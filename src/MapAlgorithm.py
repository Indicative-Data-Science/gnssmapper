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

    def generate_data(self,building):
        z = self.map.get_height(self.observations,[building]*len(self.observations))
        # ss = self.observations["ss"].fillna(0)
        ss = self.observations["ss"]
        data=np.column_stack((ss,z))
        self.data[self.key(building)]=data

    def fit_online(self,building,batch=int(4e6),iterations=1.2e7,height_lr=1e-4*np.array([0.1,100,1000,0.1]),SS_lr=1e-4*np.array([2,100,1000,2]),SGD_batch=100):
        """Iterative online fitting of the two classifiers """
        key=self.key(building)
        if key not in self.data:
            self.generate_data(building)
        data=self.data[key]
        SSLR = FourParamLogisticRegression(lr=SS_lr,batch_size=SGD_batch)
        HeightLR = FourParamLogisticRegression(lr=height_lr,batch_size=SGD_batch)
        (SSLR.a, SSLR.b, SSLR.c, SSLR.d), (HeightLR.a, HeightLR.b, HeightLR.c, HeightLR.d) = self.params.get(key,self.startingParams(building))
        
        ss_ = cycle(data[:, 0])
        heights_ = cycle(data[:, 1])
        missing_ = cycle(np.isnan(data[:, 1]))
        h_param=[]
        ss_param=[]

        for _ in np.arange(0,iterations,batch):
            ss = np.fromiter(islice(ss_,batch),float,count=batch)
            heights = np.fromiter(islice(heights_,batch),float,count=batch)
            missing= np.fromiter(islice(missing_,batch),bool,count=batch)
            ss_intersect= ss[~missing]
            heights_intersect=heights[~missing]
            y = SSLR.predict(ss_intersect)
            h_param.append(fill_array(missing,HeightLR.fit_online(heights_intersect, y)))
            y = heights_intersect>HeightLR.c
            ss_param.append(fill_array(missing,SSLR.fit_online(ss_intersect, y)))
        
        params = np.stack([np.vstack(tuple(ss_param)),np.vstack(tuple(h_param))])
        self.params[key]=[list(params[0,-1,:]),list(params[1,-1,:])]
        
        return params        

    @staticmethod
    def label_convergence(labels: np.array, preds: np.array, thres_=0.001) -> bool:
        """Determine whether the models predictions have converged."""
        thres_ = thres_ * labels.shape[0]
        count_ = sum(labels!=preds)
        return count_ < thres_

    def fit_offline(self,building):
        """Iterative offline fitting of the two classifiers til convergence."""
        key=self.key(building)
        if key not in self.data:
            self.generate_data(building)
        data=self.data[key]
        data=data[~np.isnan(data[:,1])]
        SSLR = FourParamLogisticRegression()
        HeightLR = FourParamLogisticRegression()
        (SSLR.a, SSLR.b, SSLR.c, SSLR.d), (HeightLR.a, HeightLR.b, HeightLR.c, HeightLR.d) = self.params.get(key,self.startingParams(building))
        
        ss = data[:, 0]
        heights = data[:, 1]
        param=[]
        temp_preds = SSLR.predict(ss)
        converged = False
        count=0
        while (not converged and count<100):
            HeightLR.fit_offline(heights, temp_preds)
            mdl_output = np.array([h>HeightLR.c for h in heights])
            SSLR.fit_offline(ss, mdl_output)
            mdl_output = SSLR.predict(ss)

            converged = self.label_convergence(temp_preds, mdl_output)
            temp_preds = deepcopy(mdl_output)
            param.append([[SSLR.a,SSLR.b,SSLR.c,SSLR.d],[HeightLR.a,HeightLR.b,HeightLR.c,HeightLR.d]])
            count+=1
        self.params[key]=param[-1]
        return np.array(param)  

    def fit_SS(self,building,ss_param):
        """fitting of height classifer with set signal strength"""
        key=self.key(building)
        if key not in self.data:
            self.generate_data(building)
        data=self.data[key]
        data=data[~np.isnan(data[:,1])]
        SSLR = FourParamLogisticRegression()
        HeightLR = FourParamLogisticRegression()
        ss_unused_params, (HeightLR.a, HeightLR.b, HeightLR.c, HeightLR.d) = self.params.get(key,self.startingParams(building))
        SSLR.a,SSLR.b,SSLR.c,SSLR.d = (1,10,ss_param,0)
        ss = data[:, 0]
        heights = data[:, 1]

        temp_preds=SSLR.predict(ss)
        HeightLR.fit_offline(heights, temp_preds)

        param=[[SSLR.a,SSLR.b,SSLR.c,SSLR.d],[HeightLR.a,HeightLR.b,HeightLR.c,HeightLR.d]]
        self.params[key]=param
        return np.array(param)  



    def plotModels(self,building,model="SS"):
        bins = np.linspace(0,50,50)
        LR = FourParamLogisticRegression()
        params=self.params.get(self.key(building),self.startingParams(building))
        LR.a,LR.b,LR.c,LR.d = params[0] if model=="SS" else params[1]
        preds = LR.four_param_sigmoid(bins)
        plt.plot(bins,preds)
        plt.show()

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
