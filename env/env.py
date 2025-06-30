import networkx as nx
import numpy as np
import csv
import pandas as pd
import importlib

from src.env.worker import Worker
from src.env.task import Task
from src.utils.road_network_tool import *
from src.utils.utils import *
from src.config.setting_config import setting_config
from src.config.model_config import policy_config

city_name = setting_config['city']

module_name = f"src.config.map_config.{city_name}"
module = importlib.import_module(module_name)

map_config = module.map_config


class Env:
    def __init__(self, map_setting=map_config, setting=setting_config):
        # Static properties
        self.worker_speed = setting['worker_speed']
        self.time_window = setting['time_window']
        self.distance_range = setting['distance_range']
        self.penalty_rate = setting['penalty_rate']

        self.around_task_num = setting['around_task_num']
        self.optimal_task_num = setting['optimal_task_num']
        self.around_worker_num = setting['around_worker_num']
        self.around_idle_worker_num = setting['around_idle_worker_num']

        # Load map
        self.map_file_path = get_absolute_path() + "/data/map_data/"
        self.road_net = RoadNetworkTool(self.map_file_path, map_setting['bbox'],
                                        network_type=map_setting['network_type'])

        self.bbox = map_setting['bbox']

        # Dynamic properties
        self.worker_list = []
        self.task_list = []
        self.task_upload_time_list = []

        self.idle_worker_list = []

        self.timeline = 0
        self.commission = 0

        self.update_task_num = 0

        self.done = False

    """*************************External Call*************************"""

    """in mask, 0 means no mask, 1 means mask, False means no mask, True means mask"""

    def update_task_time(self, time_update_csv):

        values = []

        with open(time_update_csv, 'r') as infile:
            time_update = csv.DictReader(infile)
            for row in time_update:
                values.append(int(row['Value']))
        self.task_upload_time_list = values

    def upload_task(self, csv_doc):

        if self.update_task_num < len(self.task_upload_time_list):
            if self.task_upload_time_list[self.update_task_num] <= self.timeline:
                task_csv = pd.read_csv(csv_doc)
                while (self.update_task_num < len(self.task_upload_time_list) and
                       self.task_upload_time_list[self.update_task_num] <= self.timeline):
                    task_inf = task_csv.loc[self.update_task_num]

                    latitude = float(task_inf['Latitude'])
                    longitude = float(task_inf['Longitude'])
                    skill = list(map(int, task_inf['Skill'].split(',')))
                    workload = int(task_inf['Working_time'])
                    expected_complete_time = int(task_inf['Expected_time'])
                    deadline = int(task_inf['Deadline'])
                    money = int(task_inf['Money'])

                    task = Task(self.update_task_num, [latitude, longitude],
                                skill, workload, expected_complete_time, deadline, money)
                    task.skill_index = task.skill.index(1)

                    task.location = self.get_nearest_point_coordinates(task.location)

                    self.task_list.append(task)

                    self.update_task_num += 1

    def upload_worker(self, worker_csv):

        with open(worker_csv, 'r') as infile:
            reader = csv.DictReader(infile)
            for index, row in enumerate(reader):
                latitude = float(row['Latitude'])
                longitude = float(row['Longitude'])
                skill = list(map(int, row['Skill'].split(',')))

                worker = Worker(
                    worker_id=index,
                    location=(latitude, longitude),
                    skill_list=skill
                )
                self.worker_list.append(worker)
                self.idle_worker_list.append(worker.id)

        self.initialize_worker_positions()

    def step(self):
        self.timeline += self.time_window

        self.update_task_worker()

        if self.update_task_num >= len(self.task_upload_time_list) and not self.task_list:
            self.done = True

    def reset(self):
        self.worker_list = []
        self.task_list = []
        self.task_upload_time_list = []

        self.idle_worker_list = []

        self.timeline = 0
        self.commission = 0

        self.update_task_num = 0

        self.done = False

    def calculate_move_time(self, worker, task):
        distance, _ = self.road_net.find_the_shortest_path(worker.location, task.location)
        return distance / self.worker_speed

    def state(self):

        w_id_list = []

        around_idle_worker_inf = []
        around_worker_inf = []
        around_task_inf = []
        optimal_task_inf = []
        self_inf = []

        around_idle_worker_mask = []
        around_worker_mask = []
        around_task_mask = []
        optimal_task_mask = []

        action_list = []

        for w in self.idle_worker_list:
            state, mask, task_id_list = self._get_w_state(self.worker_list[w])

            if task_id_list:
                around_idle_worker_inf.append(state[0])
                around_idle_worker_mask.append(mask[0])

                around_worker_inf.append(state[1])
                around_worker_mask.append(mask[1])

                around_task_inf.append(state[2])
                around_task_mask.append(mask[2])

                optimal_task_inf.append(state[3])
                optimal_task_mask.append(mask[3])

                self_inf.append(state[4])

                w_id_list.append(w)

                action_list.append(task_id_list)

        inf = [around_idle_worker_inf, around_worker_inf, around_task_inf, optimal_task_inf, self_inf]
        mask = [around_idle_worker_mask, around_worker_mask, around_task_mask, optimal_task_mask]

        return w_id_list, action_list, inf, mask

    def match(self, worker_id_list, action_list, action):
        task_id_list = []

        for i in range(len(action)):
            if action[i]:
                task_id_list.append(action_list[i][action[i] - 1])
            else:
                task_id_list.append(-1)

        for i in range(len(worker_id_list)):
            if task_id_list[i] != -1:
                self._match(task_id_list[i], worker_id_list[i])

    def vae_other_agent_action(self, worker_id_list):
        around_worker_action_list = []

        for i in range(len(worker_id_list)):
            worker = self.worker_list[worker_id_list[i]]

            around_worker_action = []

            for around_worker_id in worker.around_worker_id_list:
                match_around_worker_task_id = self.worker_list[around_worker_id].assigned_task_id
                if match_around_worker_task_id in worker.around_task_id_list:
                    around_worker_action.append(worker.around_task_id_list.index(match_around_worker_task_id) + 1)
                else:
                    around_worker_action.append(0)
            around_worker_action += [0] * (self.around_worker_num - len(around_worker_action))
            around_worker_action_list.append(around_worker_action)

        return around_worker_action_list

    def calculate_future_commission(self, worker_list):
        future_commission_list = []

        for w in worker_list:
            future_commission = self._calculate_around_future_commission(w)
            future_commission_list.append(future_commission)

        return future_commission_list


    """*************************Internal Call*************************"""

    def _calculate_around_future_commission(self, worker_id):
        worker = self.worker_list[worker_id]
        future_commission = 0

        for t_id in worker.around_task_id_list:
            for t in self.task_list:
                if t.id == t_id:
                    finish_time = t.estimated_remaining_time + t.expected_complete_time
                    if finish_time <= t.estimated_remaining_time:
                        future_commission += t.money
                    elif finish_time <= t.deadline:
                        future_commission += t.money - (finish_time - t.expected_complete_time) * self.penalty_rate

        return future_commission

    def _match(self, task_id, worker_id):
        worker = self.worker_list[worker_id]
        for t in self.task_list:
            if t.id == task_id:
                task = t

        remove_time = self.calculate_move_time(worker, task)
        worker.match_task(task_id, remove_time, task.location)
        task.match_worker(worker_id, remove_time)

        self.idle_worker_list.remove(worker_id)

        for w_id in task.assigned_worker_id:
            self.worker_list[w_id].remaining_working_time = task.estimated_remaining_time

    def _get_w_state(self, worker):
        state = []

        around_idle_worker_inf = []
        around_worker_inf = []
        around_task_inf = []
        optimal_task_inf = []

        self_inf = worker.get_state(self.bbox)

        """around idle worker"""

        distance_idle_worker = []
        idle_worker_id_list = []

        worker.around_worker_id_list = []
        worker.around_task_id_list = []

        for w in self.idle_worker_list:
            distance_of_worker = haversine(worker.location[1], worker.location[0], self.worker_list[w].location[1],
                                           self.worker_list[w].location[0])
            if distance_of_worker < self.distance_range:
                idle_worker_id_list.append(w)
                distance_idle_worker.append(distance_of_worker)

        if len(distance_idle_worker) > self.around_idle_worker_num:
            top_k_idle_worker = sorted(range(len(distance_idle_worker)),
                                       key=lambda index: distance_idle_worker[index])[
                                :self.around_idle_worker_num]
        else:
            top_k_idle_worker = sorted(range(len(distance_idle_worker)),
                                       key=lambda index: distance_idle_worker[index])

        for i in top_k_idle_worker:
            around_idle_worker_inf.append(self.worker_list[idle_worker_id_list[i]].get_state(self.bbox))

            worker.around_worker_id_list.append(idle_worker_id_list[i])

        """around not idle worker"""

        distance_working_worker = []
        working_worker_id_list = []

        for i in range(len(self.worker_list)):
            if i not in self.idle_worker_list:
                distance_of_worker = haversine(worker.location[1], worker.location[0], self.worker_list[i].location[1],
                                               self.worker_list[i].location[0])
                if distance_of_worker < self.distance_range:
                    distance_working_worker.append(distance_of_worker)
                    working_worker_id_list.append(i)

        if len(distance_working_worker) > self.around_worker_num:
            top_k_working_worker = sorted(range(len(distance_working_worker)),
                                          key=lambda index: distance_working_worker[index])[
                                   :self.around_worker_num]
        else:
            top_k_working_worker = sorted(range(len(distance_working_worker)),
                                          key=lambda index: distance_working_worker[index])
        for i in top_k_working_worker:
            around_worker_inf.append(self.worker_list[working_worker_id_list[i]].get_state(self.bbox))

        """task"""

        distance_task = []
        distance_optimal_task = []

        around_task_index = []
        optimal_task_index = []

        optimal_task_id_list = []
        sort_optimal_task_id_list = []

        for i in range(len(self.task_list)):
            distance_of_task = haversine(worker.location[1], worker.location[0], self.task_list[i].location[1],
                                         self.task_list[i].location[0])
            if distance_of_task < self.distance_range:
                if self.task_list[i].is_worker_qualified_for_task(worker):
                    distance_optimal_task.append(distance_of_task)
                    optimal_task_index.append(i)

                    optimal_task_id_list.append(self.task_list[i].id)

                else:
                    distance_task.append(distance_of_task)
                    around_task_index.append(i)

        """around task"""

        if len(distance_task) > self.around_task_num:
            top_k_task = sorted(range(len(distance_task)), key=lambda index: distance_task[index])[
                         :self.around_task_num]
        else:
            top_k_task = sorted(range(len(distance_task)), key=lambda index: distance_task[index])

        for i in top_k_task:
            around_task_inf.append(self.task_list[around_task_index[i]].get_state(self.bbox))

            worker.around_task_id_list.append(self.task_list[around_task_index[i]].id)

        """optimal task"""

        if len(distance_optimal_task) > self.optimal_task_num:
            top_k_optimal_task = sorted(range(len(distance_optimal_task)),
                                        key=lambda index: distance_optimal_task[index])[
                                 :self.optimal_task_num]
        else:
            top_k_optimal_task = sorted(range(len(distance_optimal_task)),
                                        key=lambda index: distance_optimal_task[index])

        for i in top_k_optimal_task:
            optimal_task_inf.append(self.task_list[optimal_task_index[i]].get_state(self.bbox))
            sort_optimal_task_id_list.append(optimal_task_id_list[i])

            worker.around_task_id_list.append(self.task_list[optimal_task_index[i]].id)

        state.append(around_idle_worker_inf)
        state.append(around_worker_inf)
        state.append(around_task_inf)
        state.append(optimal_task_inf)
        state.append(self_inf)

        mask = [get_mask(self.around_idle_worker_num, len(around_idle_worker_inf)),
                get_mask(self.around_worker_num, len(around_worker_inf)),
                get_mask(self.around_task_num, len(around_task_inf)),
                get_mask(self.optimal_task_num, len(optimal_task_inf))]

        state = padding(state, mask)

        return state, mask, sort_optimal_task_id_list

    def update_task_worker(self):
        # update after step
        remove_index = []
        for index, t in enumerate(self.task_list):
            if t.time_update(self.time_window):
                for w_id in t.assigned_worker_id:
                    self.worker_list[w_id].finish_update()
                    self.idle_worker_list.append(w_id)

                remove_index.append(index)

        for i in sorted(remove_index, reverse=True):
            if self.task_list[i].calculate_money(self.penalty_rate) > 0:
                self.commission += self.task_list[i].calculate_money(self.penalty_rate)
            self.task_list.pop(i)

        for w in self.worker_list:
            w.time_update(self.time_window)

    def initialize_worker_positions(self):
        for w in self.worker_list:
            w.location = self.get_nearest_point_coordinates(w.location)

    def get_nearest_point_coordinates(self, coord):
        point_id = self.road_net.find_nearest_node(coord)
        point_inf = self.road_net.graph.nodes[point_id]
        nearest_coord = (point_inf['y'], point_inf['x'])  # (latitude, longitude)
        return nearest_coord


def get_mask(max_num, actual_num):
    if actual_num == 0:
        return [0] * max_num
    mask = [0] * actual_num + [1] * (max_num - actual_num)
    return mask


def padding(state, mask):
    for i in range(len(mask)):
        if state[i]:
            state[i] += [[0] * len(state[i][0])] * (len(mask[i]) - len(state[i]))
        else:
            if i < 2:
                state[i] = [[0] * policy_config["idle_worker_input_dim"]] * len(mask[i])
            else:
                state[i] = [[0] * policy_config["around_task_input_dim"]] * len(mask[i])
    return state


if __name__ == "__main__":
    print("start")
    city_name = "CD2"

    module_name = f"src.config.map_config.{city_name}"
    module = importlib.import_module(module_name)
    map_config = module.map_config

    env = Env(map_setting=map_config)

    import matplotlib.pyplot as plt

    ox.plot_graph(env.road_net.graph, edge_linewidth=0.5, node_size=0.5, )
    plt.show()
