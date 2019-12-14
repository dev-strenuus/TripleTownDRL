import random
import numpy as np

class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, s, a, r, s2):
        # S represents current state, a is action,
        # r is reward, d is whether it is the end,
        # and s2 is next state
        experience = np.array([s, a, r, s2])
        self.buffer.append(experience)
        if self.size() > self.buffer_size:
            self.buffer.pop(0)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):

        batch = []

        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        # Maps each experience in batch in batches of states, actions, rewards
        # and new states
        s_batch, a_batch, r_batch, s2_batch = np.array(batch).T

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.buffer.clear()