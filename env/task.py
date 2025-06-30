from src.utils.utils import flatten, normalized_position

from src.config.setting_config import normalization_config


class Task:
    def __init__(self, task_id, location, skill, workload, expected_complete_time, deadline, money):
        # Static properties
        self.id = task_id
        self.location = location
        self.skill = skill
        self.skill_index = 0
        self.workload = workload
        self.expected_complete_time = expected_complete_time
        self.deadline = deadline
        self.money = money

        self.existence_time = 0

        # Dynamic properties

        self.assigned_worker_id = []
        self.not_arrived_worker_id = []

        self.not_arrived_worker_remaining_time = []

        self.assigned_worker_num = 0
        self.not_arrived_worker_num = 0

        self.working_num = 0

        self.finished = False

        self.remaining_workload = self.workload

        self.estimated_remaining_time = 0

    def is_worker_qualified_for_task(self, worker):
        if worker.skill[self.skill_index]:
            return True
        return False

    def worker_work_after_arrive(self, move_time):
        if move_time >= self.estimated_remaining_time:
            return False

    def time_update(self, past_time):
        # if self.estimated_remaining_time > 0:
        self.estimated_remaining_time -= past_time
        self.existence_time += past_time
        if self.existence_time >= self.deadline:
            self.finished = True
            return True
        self.not_arrived_worker_remaining_time = [i - past_time for i in self.not_arrived_worker_remaining_time]

        arrived_worker_index = [i for i, x in enumerate(self.not_arrived_worker_remaining_time) if x <= 0]
        self.not_arrived_worker_num -= len(arrived_worker_index)
        self.working_num += len(arrived_worker_index)

        self.remaining_workload = self.remaining_workload - self.working_num * past_time

        for i in sorted(arrived_worker_index, reverse=True):
            self.remaining_workload -= self.not_arrived_worker_remaining_time[i]

            self.not_arrived_worker_id.pop(i)

        self.not_arrived_worker_remaining_time = [x for x in self.not_arrived_worker_remaining_time if x > 0]

        if self.remaining_workload <= 0:
            self.finished = True
            return True
        return False

    def calculate_money(self, discount):
        if self.existence_time <= self.expected_complete_time:
            return self.money
        elif self.existence_time <= self.deadline:
            return self.money - self.existence_time * discount
        else:
            return 0

    def calculate_estimated_remaining_time(self):
        self.estimated_remaining_time = ((sum(self.not_arrived_worker_remaining_time) + self.remaining_workload)
                                         / self.assigned_worker_num)
        return self.estimated_remaining_time

    def match_worker(self, worker_id, move_time):
        self.assigned_worker_id.append(worker_id)
        self.not_arrived_worker_id.append(worker_id)
        self.not_arrived_worker_remaining_time.append(move_time)
        self.assigned_worker_num += 1
        self.not_arrived_worker_num += 1
        self.estimated_remaining_time = self.calculate_estimated_remaining_time()

    def get_state(self, bbox):
        # embedding size = 2+8+1+1+1+1+1 = 15
        location = normalized_position(self.location[0], self.location[1], bbox)

        state = [location, self.skill,
                 normalization(self.remaining_workload, self.estimated_remaining_time, self.assigned_worker_num,
                               self.expected_complete_time, self.deadline)]

        return flatten(state)


"""normalization function"""


def normalization(remaining_workload, estimated_remaining_time, assigned_worker_num, expected_complete_time, deadline):
    return [remaining_workload / normalization_config['remaining_workload'],
            estimated_remaining_time / normalization_config['estimated_remaining_time'],
            assigned_worker_num / normalization_config['assigned_worker_num'],
            expected_complete_time / normalization_config['expected_complete_time'],
            deadline / normalization_config['deadline']]
