"""Script to run the full simulation."""

import simulator.map as mp
import simulator.receiver as rec
from shapely.wkt import loads
import simulator.gnss as gnss
import algorithms.mapAlgo as algo
import pandas as pd
import numpy as np
import yaml

def read_config(config_path: str) -> dict:
    with open(config_path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def run_experiments(config: dict):
    for experiment,experiment_settings in config["experiments"].items():
        print("running "+experiment)
        settings = config["default_settings"]
        if not experiment_settings is None:
            for module,mod_values in experiment_settings.items():
                for k,v in mod_values.items():
                    settings[module][k]=v
        if config["run_modules"]["receiver"]:
            print("running receiver")
            run_receiver(settings)
        if config["run_modules"]["gnss"]:
            print("running gnss")
            run_gnss(settings)           
        if config["run_modules"]["model_signal"]:
            print("running model signal")
            run_model_signal(settings)
        if config["run_modules"]["mapalgo_intersections"]:
            print("running map algo intersections")
            run_mapalgo_intersections(settings)
        if config["run_modules"]["mapalgo"]:
            print("running map algo")
            run_mapalgo(settings)
    
def run_receiver(settings: dict):
    dic = settings["receiver"]
    if not check_settings_valid(dic,"receiver"):
        return None
    map_ = mp.Map(dic['map'])
    time_bound=[np.datetime64(dic['time_bound_lb'],'ns'),np.datetime64(dic['time_bound_ub'],'ns')]
    polygon = None 

    if not dic['polygon'] is None:
        with open(dic['polygon']) as f: 
            wkt_ = f.read()
        polygon = loads(wkt_)

    output = rec.ReceiverPoints()
    if dic['method'] == "point_process":
        output = rec.point_process(map_,time_bound,int(dic["num_samples"]),polygon,dic["receiver_offset"])
    if dic['method'] == "random_walk":
        output = rec.random_walk(map_,time_bound,int(dic["num_samples"]),polygon,dic["receiver_offset"],dic['avg_speed'],int(dic['sampling_rate']))

    output.to_csv(dic['filename'],index=False)

def run_gnss(settings: dict):
    dic = settings["gnss"]
    if not check_settings_valid(dic,"gnss"):
        return None
    map_ = mp.Map(dic['map'])
    points = pd.read_csv(settings['receiver']['filename'],parse_dates=['t'])
    output = gnss.observe(points,map_,dic["sslb"],dic['mu_'],dic['msr_noise'])
    output.to_csv(dic['filename'],index=False)
    
def run_model_signal(settings: dict):
    #changes the signal parameters without rerunning the fresnel calcs
    dic = settings["model_signal"]
    if not check_settings_valid(dic,"model_signal"):
        return None
    obs = pd.read_csv(settings['gnss']['filename'],parse_dates=['t'])
    obs.ss,obs.pr = gnss.model_signal(obs,dic["sslb"],dic['mu_'],dic['msr_noise'])
    obs.to_csv(dic['filename'],index=False)

def run_mapalgo_intersections(settings: dict):
    dic = settings["mapalgo_intersections"]
    if not check_settings_valid(dic,"mapalgo_intersections"):
        return None
    obs = pd.read_csv(settings['model_signal']['filename'],parse_dates=['t'])
    map_ = mp.Map(dic['map'])
    algo_ = algo.MapAlgorithm(map_,dic['filename'],obs) #this automatically calculates and saves

def run_mapalgo(settings: dict):
    return None

def check_settings_valid(dic: dict,module: str):
    expected = {"receiver": ["method","filename", "map","time_bound_lb","time_bound_ub","num_samples","polygon","receiver_offset","avg_speed","sampling_rate"],
                "gnss":     ["filename","map","sslb", "mu_","msr_noise"],
                "model_signal": ["filename","sslb", "mu_","msr_noise"],
                "mapalgo_intersections": ["map","filename"],
    }

    return all([i==j for i,j in zip(expected[module],dic.keys())])






# def gen_receiver_points(config: dict, map_location: str, check_validity=True):
#     box=Map(map_location)
#     centre_x, centre_y = box.buildings.centroid.x, box.buildings.centroid.y
#     simulation_date = config['date']
#     num_samples = int(config['num_samples'])
#     ss_sd = config['ss_sd']
#     rec_points = dict()
#     observations_ = dict()

#     # 50m bounding box polygon
#     base_box = Polygon(((centre_x - 25, centre_y + 25), (centre_x + 25, centre_y + 25),
#                         (centre_x + 25, centre_y - 25), (centre_x - 25, centre_y - 25),
#                         (centre_x - 25, centre_y + 25)))

#     # Experiment 1 Polygon
#     if config['Experiment_1']['perform']:
#         delta_x = delta_y = config['Experiment_1']['polygon_length']/2
#         thickness = config['Experiment_1']['polygon_thickness']
#         outer = Polygon((   (centre_x - delta_x, centre_y + delta_y),
#                             (centre_x + delta_x, centre_y + delta_y),
#                             (centre_x + delta_x, centre_y - delta_y),
#                             (centre_x - delta_x, centre_y - delta_y),
#                             (centre_x - delta_x, centre_y + delta_y)))

#         inners = (Polygon(( (centre_x - delta_x + thickness, centre_y + delta_y - thickness),
#                             (centre_x + delta_x - thickness, centre_y + delta_y - thickness),
#                             (centre_x + delta_x - thickness, centre_y - delta_y + thickness),
#                             (centre_x - delta_x + thickness, centre_y - delta_y + thickness),
#                             (centre_x - delta_x + thickness, centre_y + delta_y - thickness))), )
#         p = Polygon(outer.exterior.coords, [inner.exterior.coords for inner in inners])
#         rec_points_ = ReceiverEmulator(box, np.datetime64(simulation_date)).point_process(polygon_=p,
#                                                                                           num_samples=num_samples)
#         obsvs = GNSSEmulator(box, np.datetime64(simulation_date)).observe(rec_points_, ss_sd)
#         # obsvs.to_csv('exp_1.csv')
#         rec_points['Experiment_1'] = rec_points_
#         observations_['Experiment_1'] = obsvs

#     # Experiment 2
#     if config['Experiment_2']['perform']:
#         rate_ = config['Experiment_2']['sampling_rate']
#         time_bound_ = [np.timedelta64(config['Experiment_2']['start_time'], 'h'),
#                        np.timedelta64(config['Experiment_2']['interval_length'], 'h')]
#         rec_points_ = ReceiverEmulator(box, np.datetime64(simulation_date)).random_walk(polygon_=base_box,
#                                                                                         sampling_rate=rate_,
#                                                                                         time_bound=time_bound_,
#                                                                                         num_samples=num_samples)
#         obsvs = GNSSEmulator(box, np.datetime64(simulation_date)).observe(rec_points_, ss_sd)
#         rec_points['Experiment_2'] = rec_points_
#         observations_['Experiment_2'] = obsvs

#         if check_validity:
#             point_validity(base_box, rec_points_)

#     # Experiment 3
#     if config['Experiment_3']['perform']:
#         delta_x = delta_y = config['Experiment_3']['polygon_length'] / 2
#         direction = {'North': [1,1], 'East': [1,-1], 'South': [-1,-1], 'West':[-1,1]}
#         thickness = config['Experiment_3']['polygon_thickness']
#         centre_x += direction[config['Experiment_3']['orientation']][0]
#         centre_y += direction[config['Experiment_3']['orientation']][1]
#         outer = Polygon(((centre_x - delta_x, centre_y + delta_y),
#                          (centre_x + delta_x, centre_y + delta_y),
#                          (centre_x + delta_x, centre_y - delta_y),
#                          (centre_x - delta_x, centre_y - delta_y),
#                          (centre_x - delta_x, centre_y + delta_y)))

#         inners = (Polygon(((centre_x - delta_x + thickness, centre_y + delta_y - thickness),
#                            (centre_x + delta_x - thickness, centre_y + delta_y - thickness),
#                            (centre_x + delta_x - thickness, centre_y - delta_y + thickness),
#                            (centre_x - delta_x + thickness, centre_y - delta_y + thickness),
#                            (centre_x - delta_x + thickness, centre_y + delta_y - thickness))),)
#         p = Polygon(outer.exterior.coords, [inner.exterior.coords for inner in inners])
#         rec_points_ = ReceiverEmulator(box, np.datetime64(simulation_date)).point_process(polygon_=p,
#                                                                                           num_samples=num_samples)
#         obsvs = GNSSEmulator(box, np.datetime64(simulation_date)).observe(rec_points_, ss_sd)
#         rec_points['Experiment_3'] = rec_points_
#         observations_['Experiment_3'] = obsvs

#     # Experiment 4
#     if config['Experiment_4']['perform']:
#         rate_ = config['Experiment_2']['sampling_rate']
#         rec_points_ = ReceiverEmulator(box, np.datetime64(simulation_date)).random_walk(polygon_=base_box,
#                                                                                         time_bound=[np.timedelta64(config['Experiment_4']['start_time'],'h'),
#                                                                                                     np.timedelta64(1000,'h')],
#                                                                                         sampling_rate=rate_,
#                                                                                         num_samples=num_samples)
#         obsvs = GNSSEmulator(box, np.datetime64(simulation_date)).observe(rec_points_, ss_sd)
#         rec_points['Experiment_4'] = rec_points_
#         observations_['Experiment_4'] = obsvs

#         if check_validity:
#             point_validity(base_box, rec_points_)

#     return box, rec_points, observations_


if __name__ == '__main__':
    import sys
    for config_path in sys.argv[1:]:
        try:
            config_ = read_config(config_path)
            run_experiments(config_)
        except ValueError:
            print(config_path+" not valid")



