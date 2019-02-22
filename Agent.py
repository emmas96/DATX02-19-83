import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.keras as keras
import random
from collections import deque


class Agent:
    def __init__(self, num_states, num_actions):
        # Learning parameters
        self.NUM_STATES = num_states
        self.HIDDEN_LAYER_SIZE = 32
        self.NUM_ACTIONS = num_actions
        self.GAMMA = 0.8
        self.ALPHA = 0.001
        self.EPSILON_FROM = 1.0
        self.EPSILON = self.EPSILON_FROM
        self.EPSILON_TO = 0.0
        self.EPSILON_DECAY = 0.9
        self.BATCH_SIZE = 32
        self.memory = deque(maxlen=100)
        self.replays = [2, 2, 1, 2, 1, 1]
        self.counter = 0

        # Initialize model
        self.model = keras.Sequential()
        self.model.add(layers.Dense(self.HIDDEN_LAYER_SIZE,
                                    input_dim=self.NUM_STATES,
                                    activation='relu'))
        # model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        self.model.add(layers.Dense(self.NUM_ACTIONS,
                                    activation='linear'))
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.ALPHA),
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
        if self.counter < 36:
            self.counter += 1
            return self.replays[(self.counter-1) % 6]
        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.NUM_ACTIONS)
        action = self.model.predict(state)
        return np.argmax(action[0])

    # Get batch size
    def getBatchSize(self):
        return self.BATCH_SIZE
