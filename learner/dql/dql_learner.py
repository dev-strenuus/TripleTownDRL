
# List of hyper-parameters and constants
from learner.dql.dql_network import DQLNetwork
from learner.dql.replay_buffer import ReplayBuffer
from learner.learner import Learner
from tensorflow.keras.utils import *
import numpy as np

import tensorflow as tf
from tensorflow import keras

BUFFER_SIZE = 20000 #experience replay buffer size
MINIBATCH_SIZE = 64
EPSILON_DECAY = 30000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.4
C = 32 #every C iterations, the weigths of model are copied into target_model


class DQLearner(Learner):

    def __init__(self, env):
        super().__init__(env)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.network = DQLNetwork(env.num_tiles)
        self.num_tiles = env.num_tiles
        self.epsilon = INITIAL_EPSILON


    def select_action(self):
        """
        It selects an action in order to make a step in the environment.
        The action is selected according to the predicted Q values for the current state by the model.
        It returns the experience to add to the replay buffer.
        """
        grid_state = np.array([to_categorical(self.grid, num_classes=self.num_tiles)])
        storehouse_state = np.array([to_categorical(self.current_tile, num_classes=self.num_tiles).reshape(self.num_tiles)])
        curr_state = [grid_state, storehouse_state]
        predict_movement, predict_q_value = self.network.predict_movement(curr_state, self.epsilon)
        action_type = "place_current"
        position_on_the_grid = predict_movement
        if position_on_the_grid >= 36:
            action_type = "place_from_storehouse"
            position_on_the_grid -= 36

        reward, [self.grid, self.current_tile] = self.env.step(action_type, [int(position_on_the_grid/6), position_on_the_grid%6])

        new_grid_state = np.array([to_categorical(self.grid, num_classes=self.num_tiles)])
        new_storehouse_state = np.array([to_categorical(self.current_tile, num_classes=self.num_tiles).reshape(self.num_tiles)])
        new_curr_state = [new_grid_state, new_storehouse_state]
        return curr_state, predict_movement, reward, new_curr_state


    def train(self, epochs):
        iterations = 0

        for epoch in range(epochs):
            alive = True
            total_reward = 0
            while alive:

                iterations += 1

                # Slowly decay the learning rate
                if self.epsilon > FINAL_EPSILON:
                    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY

                initial_state, predicted_movement, reward, new_state = self.select_action()


                total_reward += reward
                print(total_reward)
                self.replay_buffer.add(initial_state, predicted_movement, reward, new_state)

                if self.env.is_grid_full():
                    print("Earned a total of reward equal to ", total_reward)
                    self.env.reset()
                    alive = False
                    total_reward = 0

                s_batch, a_batch, r_batch, s2_batch = self.replay_buffer.sample(MINIBATCH_SIZE)
                self.network.train(s_batch, a_batch, r_batch, s2_batch)

                if iterations % C == 0:
                    self.network.target_train()
                    print("epoch "+str(epoch)+" eps "+str(self.epsilon)+" reward "+str(total_reward))
