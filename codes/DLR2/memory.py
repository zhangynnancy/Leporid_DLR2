import numpy as np
import warnings
from collections import namedtuple


Experience = namedtuple('Experience', 'user, state, action, reward')


class RingBuffer(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class Memory(object):

    def __init__(self, memory_size):
        self.limit = memory_size  # memory size

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.user = RingBuffer(self.limit)
        self.state = RingBuffer(self.limit)
        self.action = RingBuffer(self.limit)
        self.reward = RingBuffer(self.limit)
        self.rng = np.random.RandomState(123)

    def append(self, user, state, action, reward):
        self.user.append(user)
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)

    def sample_batch_indexes(self, low, high, size):
        if high - low >= size:  # enough data
            try:
                r = range(low, high)
            except NameError:
                r = range(low, high)
            batch_idxs = self.rng.choice(r, size, replace=False)
        else:
            # print('low = ', low)
            # print('high = ', high)
            # print('size = ', size)
            warnings.warn("memory not enough data")
            batch_idxs = self.rng.random_integers(low, high - 1, size=size)

        assert len(batch_idxs) == size
        return batch_idxs

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = self.sample_batch_indexes(0, len(self.user)-1, batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < len(self.user)
        assert len(batch_idxs) == batch_size

        # create experience
        experiences = []
        for idx in batch_idxs:
            user = self.user[idx - 1]
            state = self.state[idx - 1]
            action = self.action[idx - 1]
            reward = self.reward[idx - 1]

            experiences.append(Experience(user, state, action, reward))

        assert len(experiences) == batch_size
        return experiences

    def sample_and_split(self, batch_size):  # TODO
        user, state, action, reward = [], [], [], []

        experience = self.sample(batch_size, batch_idxs=None)

        for e in experience:
            user.append(e.user)
            state.append(e.state)
            action.append(e.action)
            reward.append(e.reward)

        user = np.array(user)
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)

        return user, state, action, reward

