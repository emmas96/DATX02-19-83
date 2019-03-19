import math
import random
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions, units
import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.keras as keras
import GameEnvironment

HIDDEN_LAYER_SIZE = 100
GAMMA = 0.8
ALPHA = 0.001
EPSILON_FROM = 1.0
BOARD_SIZE_X = 20
BOARD_SIZE_Y = 10
EPSILON_TO = 0.0
EPSILON_DECAY = 0.99
BATCH_SIZE = 16
NUMSTATE = 400



class SimpleAgent(base_agent.BaseAgent):
    oldState = None
    oldScore = 0
    oldAction = None

    def __init__(self):
        self.GE = GameEnvironment.GE(1)
        base_agent.BaseAgent.__init__(self)
        # Learning parameters
        self.NUM_STATES = NUMSTATE
        self.NUM_ACTIONS = NUMSTATE
        self.EPSILON = EPSILON_FROM
        self.memory = deque(maxlen=500)
        self.counter = 0
        self.score = 0
        self.c = 0

        # Initialize model
        self.model = keras.Sequential()

        self.model.add(layers.Dense(HIDDEN_LAYER_SIZE,
                                    input_dim=NUMSTATE,
                                    activation='relu'))
        self.model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        self.model.add(layers.Dense(NUMSTATE,
                                    activation='linear'))
        self.model.compile(optimizer=tf.train.AdamOptimizer(ALPHA),
                           loss='mse',
                           metrics=['accuracy'])


    def get_action(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(NUMSTATE)
        action = self.model.predict(np.reshape(state, [1, NUMSTATE]))
        return np.argmax(action[0])

    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        if obs.first:
            cam = np.array(obs.observation.feature_minimap.camera)
            campos = self._xy_locs(cam == 1)
            if np.mean(campos, axis=0).round()[1] < 32:
                self.GE.enemyPos = (39, 42)
                self.GE.ourPos = (20, 21)
            else:
                self.GE.enemyPos = (20, 21)
                self.GE.ourPos = (39, 42)

        #larva = [unit for unit in obs.observation.feature_units
        #   if unit.unit_type == units.Zerg.Larva]
        #actions.FUNCTIONS.Train_Overlord_quick.id in obs.observation.available_actions
        action = int(input("välj action:"))
        #action = random.randint(0, 5)
        if self.counter == 0:
            self.GE.set_game_action(action, obs)
            self.counter = 0

        #return self.GE.get_game_action()
        #return actions.FUNCTIONS.move_camera()

        #if actions.FUNCTIONS.Train_Drone_quick.id in obs.observation.available_actions:
        #    return actions.FUNCTIONS.Train_Drone_quick("now")

        #if len(larva) != 0:
        #    return actions.FUNCTIONS.select_point("select", (larva[0].x, larva[0].y))
        #else:
        #    return actions.FUNCTIONS.no_op()

        return self.GE.get_game_action(obs)

    @staticmethod
    def move_to(pos):
        return actions.FUNCTIONS.Move_screen("now", pos)

    def train(self):
        mini_batch = random.sample(self.memory, self.getMemoryLength())
        for state, action, reward, next_state, done in mini_batch:
            #print(state)
            #print(action)
            #print(reward)
            #print(next_state)
            target = reward

            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(np.reshape(next_state, [1,NUMSTATE])))
            target_f = self.model.predict(np.reshape(state,[1,NUMSTATE]))
            target_f[0][action] = target
            self.model.fit(np.reshape(state,[1,NUMSTATE]), target_f, epochs=1, verbose=0)
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
