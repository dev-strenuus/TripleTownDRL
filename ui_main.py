from environment.env0 import Env0
from learner.ui_learner import UILearner

if __name__== "__main__":
    env = Env0()
    learner = UILearner(env)
    while True:
        learner.select_action()
