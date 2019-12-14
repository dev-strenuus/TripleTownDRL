from environment.env0 import Env0
from learner.dql.dql_learner import DQLearner


env = Env0()
dql_learner = DQLearner(env)
dql_learner.train(100)