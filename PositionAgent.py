import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.keras as keras
import random
from collections import deque

HIDDEN_LAYER_SIZE = 81
GAMMA = 0.8
ALPHA = 0.001
EPSILON_FROM = 1.0

EPSILON_TO = 0.0
EPSILON_DECAY = 0.9
BATCH_SIZE = 32

class Agent:
    def __init__(self, num_states, num_actions):
        # Learning parameters
        self.NUM_STATES = num_states
        self.NUM_ACTIONS = num_actions
        self.EPSILON = EPSILON_FROM
        self.memory = deque(maxlen=100)
        self.counter = 0

        # Initialize model
        self.model = keras.Sequential()
        self.model.add(layers.Dense(HIDDEN_LAYER_SIZE,
                                    input_dim=self.NUM_STATES,
                                    activation='relu'))
        # model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        self.model.add(layers.Dense(self.NUM_ACTIONS,
                                    activation='linear'))
        self.model.compile(optimizer=tf.train.AdamOptimizer(ALPHA),
                           loss='mse',
                           metrics=['accuracy'])

    # Save state to memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def getMemoryLength(self):
        return len(self.memory)

    # Update the network for a mini-batch
    def train(self):
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.EPSILON > EPSILON_TO:
            self.EPSILON *= EPSILON_DECAY

    # Get next action to preform, using Q-values
    # and the epsilon-greedy policy
    def getAction(self, state):
        if np.random.rand() <= self.EPSILON:
            while True:
                rand = random.randrange(self.NUM_ACTIONS)
                if state[0][rand] == 0:
                    return rand
        action = self.model.predict(state)
        i = np.argmin(action)
        for a in range(self.NUM_ACTIONS):
            if state[0][a] == 0 and action[0][a] > action[0][i]:
                i = a
        return i

    # Get batch size
    def getBatchSize(self):
        return BATCH_SIZE
