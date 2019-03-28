import math
import random
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions, units
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from keras.utils.generic_utils import get_custom_objects

HIDDEN_LAYER_SIZE = 40*40
GAMMA = 0.9
ALPHA = 0.001
EPSILON_FROM = 1
BOARD_SIZE_X = 40
BOARD_SIZE_Y = 10
EPSILON_TO = 0.1
EPSILON_DECAY = 0.999
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

        self.model.add(layers.Dense(NUMSTATE, activation='relu', kernel_initializer='random_uniform'))
        self.model.add(layers.Dense(NUMSTATE, activation='softmax', kernel_initializer='random_uniform'))
        # self.model.add(layers.Activation(test))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(ALPHA),
                           loss='mse',
                           metrics=['accuracy'], use_multiprocessing=True)

    def get_action(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(NUMSTATE)
        action = self.model.predict(np.reshape(state, [1, NUMSTATE]))
        self.oa += 1
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

            # Imitation learning
            if self.c < 0:
                action = beacon[0].x + BOARD_SIZE_X * beacon[0].y
                self.c += 1
                print(str(self.c))
            else:
                action = self.get_action(state)

            # Save state to memory
            if self.oldAction is not None:
                if self.reward != self.oldScore:
                    self.tmpmemory.append((self.oldState, self.oldAction, 1, state, False))
                    self.oldScore = self.reward
                else:
                    self.tmpmemory.append((self.oldState, self.oldAction, 0, state, False))
            self.oldAction = action
            self.oldState = state
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
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(np.reshape(next_state, [1, NUMSTATE])))
            target_f = self.model.predict(np.reshape(state, [1, NUMSTATE]))
            target_f[0][action] = target
            s.append(np.reshape(state, [1, NUMSTATE])[0])
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
        self.oa = 0

    def pre_processing(self, state):
        new_state = state
        new_state[np.where(state == 48)] = 0
        return new_state/317

    def save_plot_data(self, x):
        self.plot_data.append(x)

