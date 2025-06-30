from src.env.env import Env, city_name

from src.net.agent import Agent
from src.config.model_config import policy_config
from src.config.setting_config import setting_config

import csv
from tqdm import tqdm
import random

env = Env()

agent = Agent()

with open('2phi.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['j', 'phi_1', "phi_2"])

    with tqdm(range(1), desc="Training", unit="task") as pbar:
        for j in pbar:

            task_id = j
            env.reset()

            task_csv = "../../data/" + city_name + "/task/200/" + str(task_id) + ".csv"
            worker_number_index = random.randint(0, 90)
            worker_number_index = 0
            worker_csv = "../../data/" + city_name + "/worker/200/" + str(worker_number_index) + ".csv"
            task_upload_time_csv = "../../data/time/200/" + str(task_id) + ".csv"

            env.update_task_time(task_upload_time_csv)
            env.upload_worker(worker_csv)

            worker_num = len(env.worker_list)

            next_w_id_list = -1

            while not env.done:

                is_step = False

                env.upload_task(task_csv)

                if next_w_id_list == -1:
                    next_w_id_list, next_action_list, next_inf, next_mask = env.state()

                w_id_list, action_list, inf, mask = next_w_id_list, next_action_list, next_inf, next_mask

                if w_id_list:
                    action = agent.take_action(inf, mask)
                    env.match(w_id_list, action_list, action)

                    around_worker_action_list = env.vae_other_agent_action(w_id_list)
                    reward = agent.reward(inf, mask, action, around_worker_action_list)

                    env.step()
                    is_step = True

                    next_w_id_list, next_action_list, next_inf, next_mask = env.state()

                    for i in range(worker_num):
                        if i in w_id_list:
                            w_id_index = w_id_list.index(i)
                            if action[w_id_index]:
                                worker_done = 1
                            else:
                                worker_done = 0
                            env.worker_list[i].buffer.store_transition(inf[0][w_id_index],
                                                                       inf[1][w_id_index],
                                                                       inf[2][w_id_index],
                                                                       inf[3][w_id_index],
                                                                       inf[4][w_id_index],

                                                                       mask[0][w_id_index],
                                                                       mask[1][w_id_index],
                                                                       mask[2][w_id_index],
                                                                       mask[3][w_id_index],

                                                                       action[w_id_index],
                                                                       reward[w_id_index],
                                                                       worker_done)
                            if i in next_w_id_list:
                                next_w_id_list_index = next_w_id_list.index(i)
                                env.worker_list[i].buffer.store_next_transition(next_inf[0][next_w_id_list_index],
                                                                                next_inf[1][next_w_id_list_index],
                                                                                next_inf[2][next_w_id_list_index],
                                                                                next_inf[3][next_w_id_list_index],
                                                                                next_inf[4][next_w_id_list_index],

                                                                                next_mask[0][next_w_id_list_index],
                                                                                next_mask[1][next_w_id_list_index],
                                                                                next_mask[2][next_w_id_list_index],
                                                                                next_mask[3][next_w_id_list_index])
                            else:
                                env.worker_list[i].buffer.store_next_transition(
                                    [[0] * policy_config["idle_worker_input_dim"] for _ in
                                     range(setting_config["around_idle_worker_num"])],
                                    [[0] * policy_config["around_worker_input_dim"] for _ in
                                     range(setting_config["around_worker_num"])],
                                    [[0] * policy_config["around_task_input_dim"] for _ in
                                     range(setting_config["around_task_num"])],
                                    [[0] * policy_config["optimal_task_input_dim"] for _ in
                                     range(setting_config["optimal_task_num"])],

                                    [0] * policy_config["around_worker_input_dim"],

                                    [0] * setting_config["around_idle_worker_num"],
                                    [0] * setting_config["around_worker_num"],
                                    [0] * setting_config["around_task_num"],
                                    [0] * setting_config["optimal_task_num"])

                if not is_step:
                    env.step()

                    next_w_id_list, next_action_list, next_inf, next_mask = env.state()

            # reward_list_list = []
            #
            # for i in env.worker_list:
            #     buffer_num = len(i.buffer.memory["actions"])
            #     (around_idle_worker_inf_list,
            #      around_worker_inf_list,
            #      around_task_inf_list,
            #      optimal_task_inf_list,
            #      self_inf_list,
            #      around_idle_worker_mask_list,
            #      around_worker_mask_list,
            #      around_task_mask_list,
            #      optimal_task_mask_list,
            #      actions_list,
            #      rewards_list,
            #      dones_list,
            #      next_around_idle_worker_inf_list,
            #      next_around_worker_inf_list,
            #      next_around_task_inf_list,
            #      next_optimal_task_inf_list,
            #      next_self_inf_list,
            #      next_around_idle_worker_mask_list,
            #      next_around_worker_mask_list,
            #      next_around_task_mask_list,
            #      next_optimal_task_mask_list) = i.buffer.sample(buffer_num)
            #
            #     reward_list_list.append(rewards_list)
            #
            # old_phi = agent.phi
            # old_phi_2 = agent.phi_2
            #
            # agent.fine_tune(env.commission, reward_list_list)

            reward_list_list = []

            for i in env.worker_list:
                buffer_num = len(i.buffer.memory["actions"])
                (around_idle_worker_inf_list,
                 around_worker_inf_list,
                 around_task_inf_list,
                 optimal_task_inf_list,
                 self_inf_list,
                 around_idle_worker_mask_list,
                 around_worker_mask_list,
                 around_task_mask_list,
                 optimal_task_mask_list,
                 actions_list,
                 rewards_list,
                 dones_list,
                 next_around_idle_worker_inf_list,
                 next_around_worker_inf_list,
                 next_around_task_inf_list,
                 next_optimal_task_inf_list,
                 next_self_inf_list,
                 next_around_idle_worker_mask_list,
                 next_around_worker_mask_list,
                 next_around_task_mask_list,
                 next_optimal_task_mask_list) = i.buffer.sample(buffer_num)

                if around_idle_worker_inf_list:

                    # reward_list = [(i - old_phi_2) / old_phi * agent.phi + agent.phi_2 for i in rewards_list]

                    agent.implicit_gradient(around_idle_worker_inf_list,
                                            around_worker_inf_list,
                                            around_task_inf_list,
                                            optimal_task_inf_list,
                                            self_inf_list,
                                            around_idle_worker_mask_list,
                                            around_worker_mask_list,
                                            around_task_mask_list,
                                            optimal_task_mask_list,
                                            actions_list,
                                            rewards_list,
                                            dones_list,
                                            next_around_idle_worker_inf_list,
                                            next_around_worker_inf_list,
                                            next_around_task_inf_list,
                                            next_optimal_task_inf_list,
                                            next_self_inf_list,
                                            next_around_idle_worker_mask_list,
                                            next_around_worker_mask_list,
                                            next_around_task_mask_list,
                                            next_optimal_task_mask_list)

                    reward_list_list.append(rewards_list)

                    writer.writerow([j, agent.phi, agent.phi_2])

            agent.time_update(env.commission, reward_list_list)
            print(123)

