"""Script to run the full simulation."""

from receiver import *
from map import Map
from gnss import GNSSEmulator
import yaml


def read_config(config_path: str) -> dict:
    with open(config_path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def point_validity(bounded_box: Polygon, points_: ReceiverPoints):
    """Ensure all the receiver points are enclosed in the bounding box."""
    for point_ in points_.loc[:, ['x','y']].values:
        assert bounded_box.contains(Point(point_)) is True
    return


def gen_receiver_points(config: dict, map_location: str, check_validity=True):
    box=Map(map_location)
    centre_x, centre_y = box.buildings.centroid.x, box.buildings.centroid.y
    simulation_date = config['date']
    num_samples = int(config['num_samples'])
    ss_sd = config['ss_sd']
    rec_points = dict()
    observations_ = dict()

    # 50m bounding box polygon
    base_box = Polygon(((centre_x - 25, centre_y + 25), (centre_x + 25, centre_y + 25),
                        (centre_x + 25, centre_y - 25), (centre_x - 25, centre_y - 25),
                        (centre_x - 25, centre_y + 25)))

    # Experiment 1 Polygon
    if config['Experiment_1']['perform']:
        delta_x = delta_y = config['Experiment_1']['polygon_length']/2
        thickness = config['Experiment_1']['polygon_thickness']
        outer = Polygon((   (centre_x - delta_x, centre_y + delta_y),
                            (centre_x + delta_x, centre_y + delta_y),
                            (centre_x + delta_x, centre_y - delta_y),
                            (centre_x - delta_x, centre_y - delta_y),
                            (centre_x - delta_x, centre_y + delta_y)))

        inners = (Polygon(( (centre_x - delta_x + thickness, centre_y + delta_y - thickness),
                            (centre_x + delta_x - thickness, centre_y + delta_y - thickness),
                            (centre_x + delta_x - thickness, centre_y - delta_y + thickness),
                            (centre_x - delta_x + thickness, centre_y - delta_y + thickness),
                            (centre_x - delta_x + thickness, centre_y + delta_y - thickness))), )
        p = Polygon(outer.exterior.coords, [inner.exterior.coords for inner in inners])
        rec_points_ = ReceiverEmulator(box, np.datetime64(simulation_date)).point_process(polygon_=p,
                                                                                          num_samples=num_samples)
        obsvs = GNSSEmulator(box, np.datetime64(simulation_date)).observe(rec_points_, ss_sd)
        # obsvs.to_csv('exp_1.csv')
        rec_points['Experiment_1'] = rec_points_
        observations_['Experiment_1'] = obsvs

    # Experiment 2
    if config['Experiment_2']['perform']:
        rate_ = config['Experiment_2']['sampling_rate']
        time_bound_ = [np.timedelta64(config['Experiment_2']['start_time'], 'h'),
                       np.timedelta64(config['Experiment_2']['interval_length'], 'h')]
        rec_points_ = ReceiverEmulator(box, np.datetime64(simulation_date)).random_walk(polygon_=base_box,
                                                                                        sampling_rate=rate_,
                                                                                        time_bound=time_bound_,
                                                                                        num_samples=num_samples)
        obsvs = GNSSEmulator(box, np.datetime64(simulation_date)).observe(rec_points_, ss_sd)
        rec_points['Experiment_2'] = rec_points_
        observations_['Experiment_2'] = obsvs

        if check_validity:
            point_validity(base_box, rec_points_)

    # Experiment 3
    if config['Experiment_3']['perform']:
        delta_x = delta_y = config['Experiment_3']['polygon_length'] / 2
        direction = {'North': [1,1], 'East': [1,-1], 'South': [-1,-1], 'West':[-1,1]}
        thickness = config['Experiment_3']['polygon_thickness']
        centre_x += direction[config['Experiment_3']['orientation']][0]
        centre_y += direction[config['Experiment_3']['orientation']][1]
        outer = Polygon(((centre_x - delta_x, centre_y + delta_y),
                         (centre_x + delta_x, centre_y + delta_y),
                         (centre_x + delta_x, centre_y - delta_y),
                         (centre_x - delta_x, centre_y - delta_y),
                         (centre_x - delta_x, centre_y + delta_y)))

        inners = (Polygon(((centre_x - delta_x + thickness, centre_y + delta_y - thickness),
                           (centre_x + delta_x - thickness, centre_y + delta_y - thickness),
                           (centre_x + delta_x - thickness, centre_y - delta_y + thickness),
                           (centre_x - delta_x + thickness, centre_y - delta_y + thickness),
                           (centre_x - delta_x + thickness, centre_y + delta_y - thickness))),)
        p = Polygon(outer.exterior.coords, [inner.exterior.coords for inner in inners])
        rec_points_ = ReceiverEmulator(box, np.datetime64(simulation_date)).point_process(polygon_=p,
                                                                                          num_samples=num_samples)
        obsvs = GNSSEmulator(box, np.datetime64(simulation_date)).observe(rec_points_, ss_sd)
        rec_points['Experiment_3'] = rec_points_
        observations_['Experiment_3'] = obsvs

    # Experiment 4
    if config['Experiment_4']['perform']:
        rate_ = config['Experiment_2']['sampling_rate']
        rec_points_ = ReceiverEmulator(box, np.datetime64(simulation_date)).random_walk(polygon_=base_box,
                                                                                        time_bound=[np.timedelta64(config['Experiment_4']['start_time'],'h'),
                                                                                                    np.timedelta64(1000,'h')],
                                                                                        sampling_rate=rate_,
                                                                                        num_samples=num_samples)
        obsvs = GNSSEmulator(box, np.datetime64(simulation_date)).observe(rec_points_, ss_sd)
        rec_points['Experiment_4'] = rec_points_
        observations_['Experiment_4'] = obsvs

        if check_validity:
            point_validity(base_box, rec_points_)

    return box, rec_points, observations_


if __name__ == '__main__':
    conn = read_config('config.yaml')
    box, rec_points, observations = gen_receiver_points(conn, '../../map/box.txt')
    # algo_ = MapAlgorithm(box, observations)
    # offline_params = algo_.fit_offline(algo_.buildings[0])
    # online_params = algo_.fit_online(algo_.buildings[0])


