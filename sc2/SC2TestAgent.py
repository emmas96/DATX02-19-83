import math
import random
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions, units
import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.keras as keras

HIDDEN_LAYER_SIZE = 225
GAMMA = 0.8
ALPHA = 0.001
EPSILON_FROM = 1.0
BOARD_SIZE_X = 15
BOARD_SIZE_Y = 10
EPSILON_TO = 0.0
EPSILON_DECAY = 0.999
BATCH_SIZE = 16



class SimpleAgent(base_agent.BaseAgent):
    oldState = None
    oldScore = 0
    oldAction = None

    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        # Learning parameters
        self.NUM_STATES = 225
        self.NUM_ACTIONS = 225
        self.EPSILON = EPSILON_FROM
        self.memory = deque(maxlen=20000)
        self.counter = 0
        self.score = 0

        # Initialize model
        self.model = keras.Sequential()

        self.model.add(layers.Dense(HIDDEN_LAYER_SIZE,
                                    input_dim=225,
                                    activation='relu'))
        self.model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        self.model.add(layers.Dense(self.NUM_ACTIONS,
                                    activation='linear'))
        self.model.compile(optimizer=tf.train.AdamOptimizer(ALPHA),
                           loss='mse',
                           metrics=['accuracy'])

    def get_action(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.NUM_ACTIONS)
        action = self.model.predict(np.reshape(state, [1,225]))
        return np.argmax(action[0])

    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:

            marines = [unit for unit in obs.observation.feature_units
                       if unit.unit_type == units.Terran.Marine]
            beacon = [unit for unit in obs.observation.feature_units
                     if unit.unit_type == 317]
            x = abs(marines[0].x - beacon[0].x)
            y = abs(marines[0].y - beacon[0].y)
            h = math.sqrt(x ** 2 + y ** 2)
            reward = self.reward - self.score
            self.score += 1

            state = np.array(obs.observation.feature_screen.unit_type)
            action = self.get_action(state)

            if self.oldAction is not None and obs.reward == 1:
                self.memory.append((self.oldState, self.oldAction, reward, state, False))
                print("spara")

            #if self.oldAction is not None:
            #    if self.reward != self.oldScore:
            #        self.memory.append((self.oldState, self.oldAction, 1, state, False))
            #        self.oldScore = self.reward
            #    else:
            #        self.memory.append((self.oldState, self.oldAction, 0, state, False))

            self.oldAction = action
            self.oldState = state

            #player_relative = obs.observation.feature_screen.player_relative
            #x, y = (player_relative == 3).nonzero()
            #print(x[0])
            #beacon = list(zip(x,y))
            #if not beacon:
            #    return actions.FUNCTIONS.no_op()
            #beacon_center = numpy.mean(beacon, axis=0).round()

            #stuff = [unit for unit in obs.observation.feature_units
            #         if unit.unit_type == units.Terran.Marine]
            #print(stuff[0].radius)

            #marines = [unit for unit in obs.observation.feature_units
            #           if unit.unit_type == units.Terran.Marine]
            #marine = marines[0]
            #stuff = obs.observation.feature_screen.unit_type.nonzero()

            return self.move_to((action % BOARD_SIZE_X, action/BOARD_SIZE_X))
        else:
            return actions.FUNCTIONS.select_army("select")

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
                target = reward + GAMMA * np.amax(self.model.predict(np.reshape(next_state, [1,225])))
            target_f = self.model.predict(np.reshape(state,[1,225]))
            target_f[0][action] = target
            self.model.fit(np.reshape(state,[1,225]), target_f, epochs=1, verbose=0)
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

