from learner.learner import Learner


class UILearner(Learner):

    def __init__(self, env):
        super().__init__(env)

    def select_action(self):
        self.print_grid()
        print("In grid -> 0, from storehouse -> 1: ")
        choice = int(input())
        print("Insert x and y: ")
        x, y = map(int, input().split())
        reward = 0
        if choice == 0:
            reward, self.grid, self.current_tile = self.env.step("place_current", [x, y])
        else:
            reward, self.grid, self.current_tile = self.env.step("place_from_storehouse", [x, y])

