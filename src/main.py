import numpy as np
import pandas as pd
import gnss
import map
import receiver
import MapAlgorithm
import matplotlib.pyplot as plt


day=np.datetime64("2020-06-07")
box=map.Map('/Users/terry/GNSS/map/box.txt')



###code to generate observations

gnss_ =gnss.GNSSEmulator(box,day)
# rec=receiver.ReceiverEmulator(box,day)
# rec_points=rec.point_process(500000)
obs=gnss_.observe(rec_points)
# obs.to_csv("observations.csv")

### code to generate building algo data

#obs=pd.read_csv("observations.csv")
# a1=MapAlgorithm.MapAlgorithm(box,obs)
# [a1.generate_data(b) for b in a1.buildings]
# np.savez("data.npz",**a1.data)

####code to test different online set-ups



data=np.load("data.npz")
a1=MapAlgorithm.MapAlgorithm(box,None,data) #replace None with obs if needing to regenerate data

# a1.params={}
# a1.fit_offline(a1.buildings[0])
# a1.plotModels(a1.buildings[0])
# # a1.plotModels(a1.buildings[0],model="height")
# # # print(time.process_time() - start)


# # offline_height = a1.height(a1.buildings[0])
# offline_params = a1.params["0"]
# # offline_params[1][2]+1.5/offline_params[1][1]
# offline_params
# offline_height
# start = time.process_time()import cProfile

r=[]
for x in [5,15,25,35,45]:
    a1.params={}
    online_params = a1.fit_SS(a1.buildings[0],x)
    r.append(online_params)

print([x[1][2] for x in r])
