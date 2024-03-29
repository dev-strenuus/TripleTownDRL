from environment.env0 import Env0
from learner.random_learner import RandomLearner

if __name__== "__main__":
    tot_reward = 0
    num_experiments = 200
    for experiment in range(num_experiments):
        env = Env0()
        learner = RandomLearner(env)
        while env.is_grid_full() is not True:
            learner.select_action()
        tot_reward += learner.cum_reward
    print(tot_reward/num_experiments)
