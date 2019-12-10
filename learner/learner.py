class Learner:

    def __init__(self, env):
        self.env = env
        self.grid, self.current_tile = self.env.reset()
        self.cum_reward = 0

    def print_grid(self):
        print("Current tile: " + str(self.current_tile))
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                print(self.grid[i][j], end = ' ')
            print("\n")

    def select_action(self):
        return