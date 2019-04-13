import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.keras as keras
import random
from collections import deque


config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


replays = [2, 2, 1, 2, 1, 1]


class Agent:
    def __init__(self, num_states, num_actions, Gamma, Et, Mb, imi):
        # Learning parameters
        self.NUM_STATES = num_states
        self.NUM_ACTIONS = num_actions
        self.HIDDEN_LAYER_SIZE = 8
        self.ALPHA = 0.001
        self.EPSILON_FROM = 1.0
        self.EPSILON_DECAY = 0.995
        self.IMITATION = 0
        self.EPSILON_TO = Et
        self.GAMMA = Gamma
        self.BATCH_SIZE = Mb
        self.IMITATION = imi
        self.EPSILON = self.EPSILON_FROM
        self.memory = deque(maxlen=1000)
        self.counter = 0

        # Initialize model
        self.model = keras.Sequential()
        self.model.add(layers.Dense(self.HIDDEN_LAYER_SIZE,
                                    input_dim=self.NUM_STATES,
                                    activation='relu'))
        # model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        self.model.add(layers.Dense(self.NUM_ACTIONS,
                                    activation='linear'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.ALPHA),
                           loss='mse',
                           metrics=['accuracy'])

    # Save state to memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def getMemoryLength(self):
        return len(self.memory)

    # Update the network for a mini-batch
    def train(self):
        mini_batch = random.sample(self.memory, self.BATCH_SIZE)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.EPSILON > self.EPSILON_TO:
            self.EPSILON *= self.EPSILON_DECAY

    # Get next action to preform, using Q-values
    # and the epsilon-greedy policy
    def getAction(self, state):
        if self.counter < self.IMITATION:
            self.counter += 1
            return replays[(self.counter-1) % 6]
        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.NUM_ACTIONS)
        action = self.model.predict(state)
        return np.argmax(action[0])

    # Get batch size
    def getBatchSize(self):
        return self.BATCH_SIZE

    def get_epsilon(self):
        return self.EPSILON

