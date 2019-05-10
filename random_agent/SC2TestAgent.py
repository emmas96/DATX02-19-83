import math
import random
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions, units
import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.keras as keras
import random_agent.GameEnvironment as GameEnvironment

HIDDEN_LAYER_SIZE = 16

ALPHA = 0.001
EPSILON_FROM = 1.0
EPSILON_DECAY = 1
NUMSTATE = 9
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
        self.BATCH_SIZE = 256
        self.GAMMA = 0.9
        self.EPSILON_TO = 0.2
        self.imi = 0

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
        #print("ditt problem")
        action = self.model.predict(np.reshape(state, [1, NUMSTATE]))
        return np.argmax(action[0])

    def get_imi_action(self, state):
        suply_limit = state[1]
        total_suply = state[2]
        minerals = state[0]
        workers = state[4]
        army = state[5]
        nr_spawn_pools = state[6]
        nr_queens = state[7]
        amount_energy = state[8]

        action = 0

        print(f"\nNR Queens:{nr_queens}, NR Pools:{nr_spawn_pools}, Energy:{amount_energy}")

        #build overloard when supply is full
        if suply_limit - total_suply <= 1:
            #Build overloard
            action = 2
        elif minerals > 200 and nr_spawn_pools < 1:
            #Spawnpool
            action = 3
        elif workers < 16:
            #Build drone
            action = 1
        elif minerals > 150 and nr_spawn_pools >= 1 and nr_queens < 1:
            #Build queen
            action = 9
        elif nr_queens > 0 and amount_energy > 25:
            #spawn larva
            action = 10
        elif army > 20:
            #Attack
            action = 5
        else:
            #Build zergling
            action = 4

        print(f"I made the action {action}")
        return action

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
                if(self.c < self.imi):
                    action = self.get_imi_action(self.get_state(obs))
                    self.c += 1
                else:
                    action = self.get_action(self.get_state(obs))
                #print(str(action))
                state = self.get_state(obs)
                #state = self.pre_processing(state)
                if np.random.rand() <= 0.5:
                    if self.oldAction is not None:
                        if obs.observation['score_cumulative'][0] != self.oldScore:
                            self.memory.append((self.oldState, self.oldAction,
                                                obs.observation['score_cumulative'][0] - self.oldScore, state, False))

                        else:
                            self.memory.append((self.oldState, self.oldAction,
                                                obs.observation['score_cumulative'][0] - self.oldScore, state, False))
                self.oldScore = obs.observation['score_cumulative'][0]
                self.oldAction = action
                self.oldState = state

                self.GE.set_game_action(action, obs)
        return self.GE.get_game_action(obs)

    def train(self):
        mini_batch = random.sample(self.memory, self.getMemoryLength())
        for state, action, reward, next_state, done in mini_batch:
            target = reward

            if not done:
                target = reward + self.GAMMA * np.amax(self.model.predict(np.reshape(next_state, [1,NUMSTATE])))
            target_f = self.model.predict(np.reshape(state,[1,NUMSTATE]))
            target_f[0][action] = target
            self.model.fit(np.reshape(state,[1,NUMSTATE]), target_f, epochs=1, verbose=0)
        if self.EPSILON > self.EPSILON_TO:
            self.EPSILON *= EPSILON_DECAY

    def get_batch_size(self):
        return self.BATCH_SIZE

    def getMemoryLength(self):
        return len(self.memory)

    def getBatchSize(self):
        return self.BATCH_SIZE

    def reset_game(self):
        self.oldScore = 0
        self.score = 0

    def _xy_locs(self, mask):
        y, x = mask.nonzero()
        return list(zip(x, y))

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def get_state(self, obs):
        minerals = obs.observation.player[1]
        supply_limit = obs.observation.player[4]
        total_supply = obs.observation.player[3]
        army_supply = obs.observation.player[5]
        workers = obs.observation.player[6]
        army = obs.observation.player[8]
        nr_spawn_pools = len(self.get_units_by_type(obs, units.Zerg.SpawningPool))
        queens = self.get_units_by_type(obs, units.Zerg.Queen)
        if len(queens) > 0:
            amount_energy = max([q.energy for q in queens])
        else:
            amount_energy = 0

        state = (
        minerals, supply_limit, total_supply, army_supply, workers, army, nr_spawn_pools, len(queens), amount_energy)
        return state
