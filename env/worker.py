from src.utils.utils import flatten, normalized_position
from src.net.buffer import Buffer


class Worker:
    def __init__(self, worker_id, location, skill_list):

        # Static properties

        self.id = worker_id
        self.skill = skill_list

        # Dynamic properties

        self.location = location  # lat, lon
        self.assigned_task_id = None
        self.is_arrive = False

        self.remaining_arrive_time = 0

        self.remaining_working_time = 0  # todo

        self.around_task_id_list = []
        self.around_worker_id_list = []

        self.buffer = Buffer()

    def time_update(self, pass_time):

        if not self.is_arrive:
            self.remaining_arrive_time -= pass_time

            if self.remaining_arrive_time <= 0:
                self.is_arrive = True
                self.remaining_arrive_time = 0

    def match_task(self, task_id, move_time, task_location):

        self.assigned_task_id = task_id
        self.remaining_arrive_time = move_time
        self.is_arrive = False

        self.location = task_location

    def finish_update(self):

        self.assigned_task_id = None
        self.is_arrive = False

        self.remaining_arrive_time = 0
        self.remaining_arrive_time = 0

        self.around_task_id_list = []
        self.around_worker_id_list = []

    def get_state(self, bbox):

        #   if worker is idle, idle = 1, else idle = 0
        #   embedding size = 2+1+1+8 = 12

        if self.assigned_task_id is None:
            idle = 1
        else:
            idle = 0

        location = normalized_position(self.location[0], self.location[1], bbox)

        state = [location, idle, self.remaining_arrive_time, self.skill]

        return flatten(state)
