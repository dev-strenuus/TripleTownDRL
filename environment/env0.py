import numpy as np

from environment.environment import Environment


class Env0(Environment):
    """
    rules: https://support.spryfox.com/hc/en-us/articles/219104828-How-to-play-Triple-Town
    rules: https://vulcanpost.com/237711/how-to-be-a-pro-triple-town/
    """
    def __init__(self):
        self.grid_size = 6
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.triple_merge_mapping = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
        self.points = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                       8: 8}
        # self.points = {1: 5, 2: 20, 3: 100, 4: 1000, 5: 2000, 6: 3000, 7: 4000, 8: 5000} # points are realistic only up to 4 for the moment
        self.num_tiles = len(self.points) + 1
        self.cum_points = 0
        self.current_tile_type = 0

    def is_grid_full(self):
        if np.count_nonzero(self.grid) == 16:
            return True
        else:
            return False

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.cum_points = 0
        self.generate_tile()
        return self.grid, self.current_tile_type

    def get_mapping(self, num_tiles, tile_type):
        return self.triple_merge_mapping[tile_type]

    def generate_tile(self):
        """
        0 -> nothing, 1 -> grass, 2 -> bush, 3 -> tree, 4 -> hut, 5 -> house, 6 -> mansion, 7 -> castle, 8 -> floating mansion, 9 -> triple castle
        :return:
        """
        self.current_tile_type = np.random.choice([1, 2, 3, 4], 100, p=[0.6, 0.25, 0.10, 0.05])[0]

    def dfs(self, tile, x, y, visited, cleaning=False):
        if x == 0 and y == 0:
            return 0
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or visited[x][y] == 1:
            return 0
        visited[x][y] = 1
        if self.grid[x][y] != tile:
            return 0
        if cleaning:
            self.grid[x][y] = 0
        return 1 + self.dfs(tile, x - 1, y, visited, cleaning) + self.dfs(tile, x + 1, y, visited,
                                                                          cleaning) + self.dfs(tile, x, y - 1,
                                                                                               visited,
                                                                                               cleaning) + self.dfs(
            tile, x, y + 1, visited, cleaning)

    def check_merge(self, x, y, tile_type):
        res = self.dfs(tile_type, x, y, np.zeros(self.grid.shape))
        if res >= 3 and tile_type != 8:
            return True
        else:
            return False

    def place_in_grid(self, tile, x, y):
        if self.grid[x][y] != 0:
            return None
        tile_type = tile
        tot_points = 0
        self.grid[x][y] = tile_type
        while self.check_merge(x, y, tile_type):
            num_tiles = self.dfs(tile_type, x, y, np.zeros(self.grid.shape), True)
            tile_type = self.get_mapping(num_tiles, tile_type)
            tot_points += self.points[tile_type]
            self.grid[x][y] = tile_type

        if tot_points == 0:
            return self.points[self.current_tile_type]
        else:
            return tot_points

    def place_in_storehouse(self):
        if self.grid[0][0] != 0:
            return None
        self.grid[0][0] = self.current_tile_type
        return 0

    def place_from_storehouse(self, x, y):
        if self.grid[0][0] == 0:
            return None
        else:
            tile = self.grid[0][0]
            res = self.place_in_grid(tile, x, y)
            if res is None:
                return None
            else:
                self.grid[0][0] = 0
                return res

    def step(self, action_type, parameters):
        """
        :param action_type:
        :param parameters: parameters[0] = x and parameters[1] = y
        :return:
        """
        if self.is_grid_full():
            return None

        res = None
        if action_type == "place_current":
            if parameters[0] == 0 and parameters[1] == 0:
                res = self.place_in_storehouse()
            else:
                res = self.place_in_grid(self.current_tile_type, parameters[0], parameters[1])
        elif action_type == "place_from_storehouse":
            res = self.place_from_storehouse(parameters[0], parameters[1])

        if res is None:
            return -1, [self.grid, self.current_tile_type]
        else:
            self.generate_tile()
            return res, [self.grid, self.current_tile_type]

