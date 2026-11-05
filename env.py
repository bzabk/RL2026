import numpy as np


class PaperSoccerEnv:
    def __init__(self, x_length=8, y_length=10, gate_size: int = 2):

        assert x_length % 2 == 0 and y_length % 2 == 0 and gate_size % 2 == 0, "error"
        assert gate_size < y_length, "error"

        self.grid = np.zeros((y_length + 3, x_length + 1, 9), dtype=np.int8)

        # channels 0-7 represent whether specific action is posible
        # 8: is boundary

        self.agent_names = {1: "player_0", -1: "player_1"}
        self.current_player = -1
        self.x_length = x_length
        self.y_length = y_length
        self.gate_size = gate_size
        self.cur_pos = [(y_length + 3) // 2, (x_length + 1) // 2]  # starting pos
        self.grid[self.cur_pos[0], self.cur_pos[1], 8] = 1

        self.rewards = {-1: 0,
                        1: 0}
        self.terminations = {-1: False,
                             1: False}
        self.ACTION_MAP = {
            # y,x
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, 1),
            4: (1, 1),
            5: (1, 0),
            6: (1, -1),
            7: (0, -1)
        }
        self._create_boundaries()
        self._set_possible_actions()

    def grid_overview(self):
        print(self.grid[:, :, 8])

    def _create_boundaries(self):

        left = self.x_length // 2 - self.gate_size // 2
        right = self.x_length // 2 + self.gate_size // 2
        self.grid[0, :, 8] = 1
        self.grid[-1, :, 8] = 1

        self.grid[:, 0, 8] = 1
        self.grid[:, -1, 8] = 1

        self.grid[[1, -2], :left + 1, 8] = 1
        self.grid[[1, -2], right:, 8] = 1

    def _set_possible_actions(self):

        left = self.x_length // 2 - self.gate_size // 2
        right = self.x_length // 2 + self.gate_size // 2

        # left band without corners
        self.grid[2:-2, 0, [0, 1, 5, 6, 7, 8]] = 1
        # right band with corners
        self.grid[2:-2, -1, [1, 2, 3, 4, 5, 8]] = 1

        # left upper corner
        self.grid[1, 0, [0, 1, 2, 3, 5, 6, 7, 8]] = 1
        # right upper corner
        self.grid[1, -1, [0, 1, 2, 3, 4, 5, 7, 8]] = 1
        # left down corner
        self.grid[-2, 0, [0, 1, 3, 4, 5, 6, 7, 8]] = 1
        # right down corner
        self.grid[-2, -1, [1, 2, 3, 4, 5, 6, 7, 8]] = 1

        self.grid[1, 1:left, [0, 1, 2, 3, 7, 8]] = 1
        self.grid[1, right + 1:-1, [0, 1, 2, 3, 7, 8]] = 1
        self.grid[-2, 1:left, [3, 4, 5, 6, 7, 8]] = 1
        self.grid[-2, right + 1:-1, [3, 4, 5, 6, 7, 8]] = 1

        self.grid[1, left, [0, 1, 7, 8]] = 1
        self.grid[1, right, [1, 2, 3, 8]] = 1
        self.grid[-2, left, [5, 6, 7, 8]] = 1
        self.grid[-2, right, [3, 4, 5, 8]] = 1

        self.grid[0, left + 1:right, [0, 1, 2, 3, 7, 8]] = 1
        self.grid[-1, left + 1:right, [3, 4, 5, 6, 7, 8]] = 1

        # unavailable areas, (for deep learning purposes?)
        self.grid[0, :left, :] = 1
        self.grid[0, right + 1:, :] = 1
        self.grid[-1, :left, :] = 1
        self.grid[-1, right + 1:, :] = 1

        # rear corners
        self.grid[0, left, [0, 1, 2, 3, 5, 6, 7, 8]] = 1
        self.grid[0, right, [0, 1, 2, 3, 4, 5, 7, 8]] = 1

        self.grid[-1, left, [0, 1, 3, 4, 5, 6, 7, 8]] = 1
        self.grid[-1, right, [1, 2, 3, 4, 5, 6, 7, 8]] = 1

    def reset(self):
        self.grid = np.zeros((self.y_length + 3, self.x_length + 1, 9), dtype=np.int8)
        self._create_boundaries()
        self._set_possible_actions()

        self.current_player = -1
        self.cur_pos = [(self.y_length + 3) // 2, (self.x_length + 1) // 2]  # starting pos
        self.rewards = {-1: 0,
                        1: 0}
        self.terminations = {-1: False,
                             1: False}

        self.grid[self.cur_pos[0], self.cur_pos[1], 8] = 1  # starting point as a bouncing point
        return self.grid.copy(), self.current_player

    def step(self, action):
        # possible actions: {0,1,2,3,4,5,6,7}
        #  0           1              2
        #
        #
        #
        # 7      starting_point       3
        #
        #
        # 6            5              4
        end = False
        rewards = {1: 0,
                   -1: 0}
        y, x = self.cur_pos
        if self.grid[y, x, action] == 1:  # action is unavailable
            self.terminations = {agent: True for agent in self.agent_names}
            rewards[self.current_player] = -1
            rewards[-self.current_player] = 1
            end = True
            return self.grid.copy(), rewards, end, self.current_player

        dy, dx = self.ACTION_MAP[action]
        self.grid[y, x, action] = 1
        new_y, new_x = y + dy, x + dx
        self.cur_pos = [new_y, new_x]
        oppposite_action = (action - 4) % 8
        self.grid[new_y, new_x, oppposite_action] = 1

        if new_y == 0:
            end = True
            self.terminations = {agent: True for agent in self.agent_names}
            rewards[-1] = -1
            rewards[1] = 1
        elif new_y == self.grid.shape[0] - 1:
            end = True
            self.terminations = {agent: True for agent in self.agent_names}
            rewards[-1] = 1
            rewards[1] = -1
        elif np.all(self.grid[new_y, new_x, 0:8] == 1):
            end = True
            self.terminations = {agent: True for agent in self.agent_names}
            rewards[self.current_player] = -1
            rewards[-self.current_player] = 1
        else:
            if self.grid[new_y, new_x, 8] == 0:
                self.current_player *= (-1)
                self.grid[new_y, new_x, 8] = 1

        return self.grid.copy(), rewards, end, self.current_player


if __name__ == "__main__":
    env = PaperSoccerEnv(x_length=8, y_length=10, gate_size=2)
    env.grid_overview()
