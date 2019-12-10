from learner.learner import Learner
from learner.random_learner import RandomLearner
import numpy as np


class GreedyLearner(RandomLearner):

    def __init__(self, env):
        super().__init__(env)


    def select_action(self):
        shuffle_x = np.arange(self.grid.shape[0])
        shuffle_y = np.arange(self.grid.shape[1])
        np.random.shuffle(shuffle_x)
        np.random.shuffle(shuffle_y)
        for x in shuffle_x:
            for y in shuffle_y:
                if self.grid[x][y] == 0 and self.env.check_merge(self.current_tile, x, y):
                    reward, self.grid, self.current_tile = self.env.step("place_current", [x, y])
                    self.cum_reward += reward
                    return

        if self.grid[0][0] != 0:
            for x in shuffle_x:
                for y in shuffle_y:
                    if self.grid[x][y] == 0 and self.env.check_merge(self.grid[0][0], x, y):
                        reward, self.grid, self.current_tile = self.env.step("place_from_storehouse", [x, y])
                        self.cum_reward += reward
                        return
        for x in shuffle_x:
            for y in shuffle_y:
                if self.grid[x][y] == self.current_tile:
                    if x-1 >= 0 and self.grid[x-1][y] == 0:
                        reward, self.grid, self.current_tile = self.env.step("place_current", [x-1, y])
                        self.cum_reward += reward
                        return
                    if x+1 < self.grid.shape[0] and self.grid[x+1][y] == 0:
                        reward, self.grid, self.current_tile = self.env.step("place_current", [x+1, y])
                        self.cum_reward += reward
                        return
                    if y-1 >= 0 and self.grid[x][y-1] == 0:
                        reward, self.grid, self.current_tile = self.env.step("place_current", [x, y-1])
                        self.cum_reward += reward
                        return
                    if y+1 < self.grid.shape[1] and self.grid[x][y+1] == 0:
                        reward, self.grid, self.current_tile = self.env.step("place_current", [x, y+1])
                        self.cum_reward += reward
                        return

        super().select_action()
