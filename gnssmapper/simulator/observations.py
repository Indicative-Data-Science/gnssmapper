""" 
=========================================
Defines a class of GNSS observations to be used in a 3D mapping algorithm 
=========================================

"""

class Positions(pd.DataFrame):
    def __init__(self,*kwargs):
        a= pd.DataFrame(*kwargs)

    @property
    def _constructor(self):
        return Positions


class Signals(pd.DataFrame):



class Observations:
    def __init__(self,positions=None,signals=None):
        self.positions = 
        self.signals =
    





def read(filename:str)->Obs:
    """Reads a gnsslogger csv file and returns an Observation object.
    """
    obs_no_sat = _read_gnsslogger(filename)
    obs=locate_satellite(obs_no_sat)  #need to change 


