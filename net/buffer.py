import numpy as np


class Buffer:
    def __init__(self):
        self.memory = {
            "around_idle_worker_inf": [],
            "around_worker_inf": [],
            "around_task_inf": [],
            "optimal_task_inf": [],
            "self_inf": [],
            "around_idle_worker_mask": [],
            "around_worker_mask": [],
            "around_task_mask": [],
            "optimal_task_mask": [],

            "actions": [],
            "rewards": [],
            "dones": [],

            "next_around_idle_worker_inf": [],
            "next_around_worker_inf": [],
            "next_around_task_inf": [],
            "next_optimal_task_inf": [],
            "next_self_inf": [],
            "next_around_idle_worker_mask": [],
            "next_around_worker_mask": [],
            "next_around_task_mask": [],
            "next_optimal_task_mask": [],
        }
        self.position = 0

    def reset(self):
        self.memory = {
            "around_idle_worker_inf": [],
            "around_worker_inf": [],
            "around_task_inf": [],
            "optimal_task_inf": [],
            "self_inf": [],
            "around_idle_worker_mask": [],
            "around_worker_mask": [],
            "around_task_mask": [],
            "optimal_task_mask": [],

            "actions": [],
            "rewards": [],
            "dones": [],

            "next_around_idle_worker_inf": [],
            "next_around_worker_inf": [],
            "next_around_task_inf": [],
            "next_optimal_task_inf": [],
            "next_self_inf": [],
            "next_around_idle_worker_mask": [],
            "next_around_worker_mask": [],
            "next_around_task_mask": [],
            "next_optimal_task_mask": [],

        }
        self.position = 0

    def store_transition(self,
                         around_idle_worker_inf,
                         around_worker_inf,
                         around_task_inf,
                         optimal_task_inf,
                         self_inf,
                         around_idle_worker_mask,
                         around_worker_mask,
                         around_task_mask,
                         optimal_task_mask,
                         action,
                         reward,
                         done):
        self.memory["around_idle_worker_inf"].append(around_idle_worker_inf)
        self.memory["around_worker_inf"].append(around_worker_inf)
        self.memory["around_task_inf"].append(around_task_inf)
        self.memory["optimal_task_inf"].append(optimal_task_inf)
        self.memory["self_inf"].append(self_inf)
        self.memory["around_idle_worker_mask"].append(around_idle_worker_mask)
        self.memory["around_worker_mask"].append(around_worker_mask)
        self.memory["around_task_mask"].append(around_task_mask)
        self.memory["optimal_task_mask"].append(optimal_task_mask)

        self.memory["actions"].append(action)
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)

        self.position += 1

    def store_next_transition(self,
                              next_around_idle_worker_inf,
                              next_around_worker_inf,
                              next_around_task_inf,
                              next_optimal_task_inf,
                              next_self_inf,
                              next_around_idle_worker_mask,
                              next_around_worker_mask,
                              next_around_task_mask,
                              next_optimal_task_mask):
        self.memory["next_around_idle_worker_inf"].append(next_around_idle_worker_inf)
        self.memory["next_around_worker_inf"].append(next_around_worker_inf)
        self.memory["next_around_task_inf"].append(next_around_task_inf)
        self.memory["next_optimal_task_inf"].append(next_optimal_task_inf)
        self.memory["next_self_inf"].append(next_self_inf)
        self.memory["next_around_idle_worker_mask"].append(next_around_idle_worker_mask)
        self.memory["next_around_worker_mask"].append(next_around_worker_mask)
        self.memory["next_around_task_mask"].append(next_around_task_mask)
        self.memory["next_optimal_task_mask"].append(next_optimal_task_mask)

    def sample(self, batch_size):
        if batch_size > self.position:
            return [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        start_idx = np.random.randint(0, self.position - batch_size + 1)

        # Extract the segment from both buffers
        around_idle_worker_inf_list = self.memory["around_idle_worker_inf"][start_idx:start_idx + batch_size]
        around_worker_inf_list = self.memory["around_worker_inf"][start_idx:start_idx + batch_size]
        around_task_inf_list = self.memory["around_task_inf"][start_idx:start_idx + batch_size]
        optimal_task_inf_list = self.memory["optimal_task_inf"][start_idx:start_idx + batch_size]
        self_inf_list = self.memory["self_inf"][start_idx:start_idx + batch_size]
        around_idle_worker_mask_list = self.memory["around_idle_worker_mask"][start_idx:start_idx + batch_size]
        around_worker_mask_list = self.memory["around_worker_mask"][start_idx:start_idx + batch_size]
        around_task_mask_list = self.memory["around_task_mask"][start_idx:start_idx + batch_size]
        optimal_task_mask_list = self.memory["optimal_task_mask"][start_idx:start_idx + batch_size]

        actions_list = self.memory["actions"][start_idx:start_idx + batch_size]
        rewards_list = self.memory["rewards"][start_idx:start_idx + batch_size]
        dones_list = self.memory["dones"][start_idx:start_idx + batch_size]

        next_around_idle_worker_inf_list = self.memory["next_around_idle_worker_inf"][start_idx:start_idx + batch_size]
        next_around_worker_inf_list = self.memory["next_around_worker_inf"][start_idx:start_idx + batch_size]
        next_around_task_inf_list = self.memory["next_around_task_inf"][start_idx:start_idx + batch_size]
        next_optimal_task_inf_list = self.memory["next_optimal_task_inf"][start_idx:start_idx + batch_size]

        next_self_inf_list = self.memory["next_self_inf"][start_idx:start_idx + batch_size]

        next_around_idle_worker_mask_list = self.memory["next_around_idle_worker_mask"][
                                            start_idx:start_idx + batch_size]
        next_around_worker_mask_list = self.memory["next_around_worker_mask"][start_idx:start_idx + batch_size]
        next_around_task_mask_list = self.memory["next_around_task_mask"][start_idx:start_idx + batch_size]
        next_optimal_task_mask_list = self.memory["next_optimal_task_mask"][start_idx:start_idx + batch_size]

        return (around_idle_worker_inf_list,
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
