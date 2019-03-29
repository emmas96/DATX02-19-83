import math
import random
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions, units
import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.keras as keras
import agent.GameEnvironment as GameEnvironment

HIDDEN_LAYER_SIZE = 16
GAMMA = 0.9
ALPHA = 0.001
EPSILON_FROM = 1.0
EPSILON_TO = 0.1
EPSILON_DECAY = 0.99
BATCH_SIZE = 128
NUM_STATE = 8
NUM_ACTIONS = 6

class BetaStar(base_agent.BaseAgent):
    oldState = None
    oldScore = 0
    oldAction = None

    def __init__(self):
        self.GE = GameEnvironment.GE(1)
        base_agent.BaseAgent.__init__(self)
        # Learning parameters
        self.EPSILON = EPSILON_FROM
        self.memory = deque(maxlen=5000)
        self.counter = 0
        self.score = 0

        # Initialize model
        self.model = keras.Sequential()
        self.model.add(layers.Dense(NUM_STATE, input_dim=NUM_STATE, activation='relu'))
        self.model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        self.model.add(layers.Dense(NUM_ACTIONS, activation='linear'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(ALPHA),
                           loss='mse',
                           metrics=['accuracy'])


    def get_action(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(NUM_ACTIONS)
        action = self.model.predict(state)
        return np.argmax(action[0])

    def step(self, obs):
        super(BetaStar, self).step(obs)
        if obs.first:
            cam = np.array(obs.observation.feature_minimap.camera)
            campos = self._xy_locs(cam == 1)
            if np.mean(campos, axis=0).round()[1] < 32:
                self.GE.enemyPos = (39, 42)
                self.GE.ourPos = (20, 21)
            else:
                self.GE.enemyPos = (20, 21)
                self.GE.ourPos = (39, 42)

        action = random.randint(0, 5)
        if self.counter == 0:
            self.GE.set_game_action(action, obs)
            self.counter = 0

        return self.GE.get_game_action(obs)

    @staticmethod
    def move_to(pos):
        return actions.FUNCTIONS.Move_screen("now", pos)

    def train(self):
        mini_batch = random.sample(self.memory, self.getMemoryLength())
        s = []
        t = []
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(next_state))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            s.append(state)
            t.append(target_f[0])
        sa = np.asarray(s)
        ta = np.asarray(t)

        # train on mini-batch
        self.model.fit(sa, ta, epochs=1, verbose=0)
        if self.EPSILON > EPSILON_TO:
            self.EPSILON *= EPSILON_DECAY

    def get_batch_size(self):
        return BATCH_SIZE

    def getMemoryLength(self):
        return len(self.memory)

    def getBatchSize(self):
        return BATCH_SIZE

    def reset_game(self):
        self.oldScore = 0
        self.score = 0

    def _xy_locs(self, mask):
        y, x = mask.nonzero()
        return list(zip(x, y))
