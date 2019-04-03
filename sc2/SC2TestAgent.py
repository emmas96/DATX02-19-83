import math
import random
import time
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions, units
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from keras.utils.generic_utils import get_custom_objects

HIDDEN_LAYER_SIZE = 40*40
GAMMA = 0
ALPHA = 0.0005
EPSILON_FROM = 1.0
BOARD_SIZE_X = 40
BOARD_SIZE_Y = 10
EPSILON_TO = 0.2
EPSILON_DECAY = 0.9995
BATCH_SIZE = 128
NUMSTATE = 40*40

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def test(x):
    return 1 / (1 + np.e**(-x / 100))


class SimpleAgent(base_agent.BaseAgent):
    oldState = None
    oldScore = 0
    oldAction = None

    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        # Learning parameters
        self.NUM_STATES = NUMSTATE
        self.NUM_ACTIONS = NUMSTATE
        self.EPSILON = EPSILON_FROM
        self.memory = deque(maxlen=200)
        self.tmpmemory = deque(maxlen=50)
        self.plot_data = deque()
        self.counter = 0
        self.score = 0
        self.c = 0
        self.oa = 0

        get_custom_objects().update({'test': layers.Activation(test)})
        # Initialize model
        self.model = keras.Sequential()

        #self.model.add(layers.Conv2D(filters=16,kernel_size=5, activation='relu', padding="same" , input_shape=(40,40,1) ))
        #self.model.add(layers.Conv2D(filters=32, kernel_size=3, padding="same", activation='relu'))
        #self.model.add(layers.BatchNorm)
        #self.model.add(layers.Flatten())
        #self.model.add(tf.keras.layers.BatchNormalization())
        #self.model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='softmax', kernel_initializer='random_uniform'))
        self.model.add(layers.Dense(NUMSTATE, activation='relu', kernel_initializer='random_uniform'))
        #self.model.add(layers.Dense(HIDDEN_LAYER_SIZE, activation='relu', kernel_initializer='random_uniform'))
        self.model.add(layers.Dense(NUMSTATE, activation='linear', kernel_initializer='random_uniform'))
        # self.model.add(layers.Activation(test))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(ALPHA),
                           loss='mse',
                           metrics=['accuracy'], use_multiprocessing=True)

    def get_action(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(NUMSTATE)
        action = self.model.predict(np.reshape(state, [1, NUMSTATE]))
        # Randomly pick locations near the prodiction
        #if np.random.rand() <= self.EPSILON:
        #    action += (random.randrange(6)-3) * 40
        #    action += (random.randrange(6) - 3)
        #plot = np.reshape(action, [40,40])
        #plt.imshow(plot, cmap='hot', interpolation='nearest')
        #plt.show()
        self.oa += 1
        #action = self.model.predict(np.expand_dims(np.expand_dims(state, axis=0),axis=3))
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
            state = self.pre_processing(state)

            if(self.c < 000):

                action = beacon[0].x + BOARD_SIZE_X * beacon[0].y
                self.c += 1
                print(str(self.c))
            else:
                action = self.get_action(state)

            #self.tmpmemory.append((self.oldState, self.oldAction,self.reward, state, False))
                #print("spara")

            if self.oldAction is not None and self.reward > 0:
                if self.reward != self.oldScore:
                    self.memory.append((self.oldState, self.oldAction, 1, state, False))
                    self.oldScore = self.reward
                else:
                    self.memory.append((self.oldState, self.oldAction, 0, state, False))

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

            #return self.move_to((action % BOARD_SIZE_X, action/BOARD_SIZE_X))
            return self.move_to((action % BOARD_SIZE_X, action / BOARD_SIZE_X))
        else:
            return actions.FUNCTIONS.select_army("select")

    @staticmethod
    def move_to(pos):
        return actions.FUNCTIONS.Move_screen("now", pos)


    def train(self):
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        s = []
        t = []

        for state, action, reward, next_state, done in mini_batch:
            #print(state)
            #print(action)
            #print(reward)
            #print(next_state)
            target = reward

            a = np.expand_dims(next_state, axis=0)
            #print(str(state.shape))
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(np.reshape(next_state, [1, NUMSTATE])))

                #target = reward + GAMMA * np.amax(self.model.predict(np.copy(np.expand_dims(a, axis=3))))
            target_f = self.model.predict(np.reshape(state, [1, NUMSTATE]))
            #target_f[0][action] = target
            #target_f = self.model.predict(np.copy(np.expand_dims(a, axis=3)))
            #target_f[0][action] = target_f[0][action]*(0.5 + reward)
            target_f[0][action] = target
            s.append(np.reshape(state, [1, NUMSTATE])[0])
            t.append(target_f[0])
        sa = np.asarray(s)
        ta = np.asarray(t)
        self.model.fit(sa, ta, epochs=1, verbose=0)
            #b = np.expand_dims(state, axis=0)
            #self.model.fit(np.expand_dims(b, axis=3), target_f, epochs=1, verbose=0)

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
        self.oa = 0

    def pre_processing(self, state):
        new_state = state
        new_state[np.where(state == 48)] = 0
        return new_state/317

    def save_plot_data(self, x):
        self.plot_data.append(x)

