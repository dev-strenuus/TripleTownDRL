import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import *

DECAY_RATE = 0.975
NUM_ACTIONS = 72 #if action a < 36, place the new tile in pos (a/6, a%6), else place tile in storehouse in position ((a-36)/6, (a-36)%6)

class DQLNetwork():
    def __init__(self, num_tiles):
        self.num_tiles = num_tiles
        self.construct_q_network()

    def define_model(self):
        input_layer1 = Input(shape=(6, 6, 9))
        input_layer2 = Input(shape=(9))
        dense1 = Dense(32, activation="sigmoid")(input_layer1)
        flattened1 = Flatten()(dense1)
        conv1 = Conv2D(filters=4, kernel_size=(2, 2), padding="same", activation="relu")(input_layer1)
        conv2 = Conv2D(filters=4, kernel_size=(3, 3), activation="relu")(conv1)
        pool = MaxPooling2D(pool_size=2, strides=1, padding='same')(conv2)
        flattened2 = Flatten()(pool)
        conc = concatenate([flattened1, flattened2, input_layer2])
        dense2 = Dense(512, activation="relu")(conc)
        output = Dense(NUM_ACTIONS)(dense2)
        model = Model([input_layer1, input_layer2], output)
        model.compile(loss='mse', optimizer='adam')
        return model

    def construct_q_network(self):

        self.model = self.define_model()
        self.target_model = self.define_model() #copy of the main model, every C iterations it gets the same weigth of the main model
        self.target_model.set_weights(self.model.get_weights())

        print("Successfully constructed networks.")


    def predict_movement(self, data, epsilon):
        """
        eps-greedy strategy, with probability eps it selects a random action.
        With probability 1-eps it selects the action that maximize the estimated Q value for the state
        """
        q_actions = self.model.predict(data)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            if rand_val < 0.1:
                opt_policy = np.random.randint(36, NUM_ACTIONS)
            else:
                opt_policy = np.random.randint(0, 36)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, s2_batch):
        """
        Training on a given batch.
        See https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/
        """
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))
        s1 = []
        s2 = []
        for i in range(batch_size):
            s1.append(s_batch[i][0][0])
            s2.append(s_batch[i][1][0])
            targets[i] = self.model.predict([[s_batch[i][0][0]], [s_batch[i][1][0]]])
            fut_action = self.target_model.predict([[s2_batch[i][0][0]], [s2_batch[i][1][0]]])
            targets[i, a_batch[i]] = r_batch[i]
            targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)
        loss = self.model.fit([s1,s2], targets)


    def save_network(self):
        self.model.save(self.checkpoint_path)

    def load_network(self):
        self.model.load_weigths(self.checkpoint_path)

    def target_train(self):
        """
        Copy the weights of model in target_model.
        """
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)