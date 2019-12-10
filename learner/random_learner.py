from learner.learner import Learner
import numpy as np


class RandomLearner(Learner):

    def __init__(self, env):
        super().__init__(env)

    def select_action(self):
        x = np.random.randint(0, 6)
        y = np.random.randint(0, 6)
        while self.grid[x][y] != 0:
            x = np.random.randint(0, 6)
            y = np.random.randint(0, 6)
        reward = 0
        if np.random.uniform() > 0.1:
            reward, self.grid, self.current_tile = self.env.step("place_current", [x, y])
        else:
            reward, self.grid, self.current_tile = self.env.step("place_from_storehouse", [x, y])
        self.cum_reward += reward