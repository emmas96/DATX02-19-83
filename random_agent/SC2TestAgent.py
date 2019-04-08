import math
import random
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions, units
import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.keras as keras
import GameEnvironment as GameEnvironment

HIDDEN_LAYER_SIZE = 16
GAMMA = 0.9
ALPHA = 0.001
EPSILON_FROM = 1.0
EPSILON_TO = 0.2
EPSILON_DECAY = 0.99
BATCH_SIZE = 256
NUMSTATE = 6
NUMACTION = 11


class SimpleAgent(base_agent.BaseAgent):
    oldState = None
    oldScore = 0
    oldAction = None

    def __init__(self):
        self.GE = GameEnvironment.GE(1)
        base_agent.BaseAgent.__init__(self)
        # Learning parameters
        self.NUM_STATES = NUMSTATE
        self.NUM_ACTIONS = NUMACTION
        self.EPSILON = EPSILON_FROM
        self.memory = deque(maxlen=5000)
        self.tmpMemory = deque(maxlen=5000)
        self.counter = 0
        self.score = 0
        self.c = 0

        # Initialize model
        self.model = keras.Sequential()

        self.model.add(layers.Dense(HIDDEN_LAYER_SIZE,
                                    input_dim=NUMSTATE,
                                    activation='relu'))
        self.model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        self.model.add(layers.Dense(NUMACTION,
                                    activation='linear'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(ALPHA),
                           loss='mse',
                           metrics=['accuracy'])


    def get_action(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(NUMACTION)
        print("ditt problem")
        action = self.model.predict(np.reshape(state, [1, NUMSTATE]))
        return np.argmax(action[0])

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        if obs.first:
            cam = np.array(obs.observation.feature_minimap.camera)
            campos = self._xy_locs(cam == 1)
            if np.mean(campos, axis=0).round()[1] < 32:
                self.GE.enemyPos = (39, 44)
                self.GE.ourPos = (22, 23)
                self.GE.overlordPlace = (0, 0)
                self.GE.enemyExp = (15, 48)
                self.GE.NatExp = (41, 20)
            else:
                self.GE.enemyPos = (19, 21)
                self.GE.ourPos = (36, 45)
                self.GE.overlordPlace = (63, 63)
                self.GE.enemyExp = (41, 20)
                self.GE.NatExp = (15, 48)

        if False: #true = controlled with console
            if len(self.GE.ActionQueue) == 0:
                i_action = input()
                if i_action in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"):
                    act = int(i_action)
                else:
                    act = 0
                self.GE.set_game_action(act,obs)
            return self.GE.get_game_action(obs)


        if self.counter == 0:
            self.counter = 0
            if len(self.GE.ActionQueue) == 0:
                action = self.get_action(self.get_state(obs))
                print(str(action))
                state = self.get_state(obs)
                #state = self.pre_processing(state)
                if self.oldAction is not None:
                    if self.reward != self.oldScore:
                        self.tmpMemory.append((self.oldState, self.oldAction, self.reward - self.oldScore, state, False))
                        self.oldScore = self.reward
                    else:
                        self.tmpMemory.append((self.oldState, self.oldAction, self.reward - self.oldScore, state, False))

                self.oldAction = action
                self.oldState = state

                self.GE.set_game_action(action, obs)
        return self.GE.get_game_action(obs)

    def train(self):
        mini_batch = random.sample(self.memory, self.getMemoryLength())
        for state, action, reward, next_state, done in mini_batch:
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

    def get_state(self, obs):
        minerals = obs.observation.player[1]
        supply_limit = obs.observation.player[4]
        total_supply = obs.observation.player[3]
        army_supply = obs.observation.player[5]
        workers = obs.observation.player[6]
        army = obs.observation.player[8]

        state = (minerals, supply_limit, total_supply, army_supply, workers, army)
        return state
