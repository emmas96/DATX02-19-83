from pysc2.lib import actions, units, features
import numpy as np
from collections import deque


class GE:

    def __init__(self, x):
        self.Num = x
        self.ActionQueue = deque(maxlen=100)

    def step(self, obs):
        super(ZergAgent, self).step(obs)

        drones = [unit for unit in obs.observation.feature_units
                  if unit.unit_type == units.Zerg.Drone]

    @staticmethod
    def get_beacon_position(obs):
        beacon = [unit for unit in obs.observation.feature_units
                  if unit.unit_type == 317]
        x = beacon[0].x
        y = beacon[0].y
        return x, y

    @staticmethod
    def move_to(pos):
        return actions.FUNCTIONS.Move_screen("now", pos)

    def set_game_action(self, action, position, obs):

        if action == 0:
            self.ActionQueue.append(actions.FUNCTIONS.no_op())
        if action == 1:
            larva = [unit for unit in obs.observation.feature_units
                     if unit.unit_type == units.Zerg.Larva]
            self.ActionQueue.append(actions.FUNCTIONS.select_point("select", (larva[0].x, larva[0].y)))
            self.ActionQueue.append(actions.FUNCTIONS.Train_Drone_quick("now"))
        if action == 2:
                self.ActionQueue.append(actions.FUNCTIONS.select_point("select", (larva[0].x, larva[0].y)))
                self.ActionQueue.append(actions.FUNCTIONS.Train_Overlord_quick("now"))
        if action == 3: #select random drone and build spawning pool, check tstarbot if you can find coding
            if len(drones) > 0:
                drone = random.choice(drones)
                self.ActionQueue.append(actions.FUNCTIONS.select_point("select", (drone.x, drone.y)))
                if (actions.FUNCTIONS.Build_SpawningPool_screen.id in
                        obs.observation.available_actions):
                        x = random.randint(0, 83)
                        y = random.randint(0, 83)
                        self.ActionQueue.append(actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y)))

        if action == 4:
            if (actions.FUNCTIONS.Train_Zergling_quick.id in
          obs.observation.available_actions):
                larva = [unit for unit in obs.observation.feature_units
                         if unit.unit_type == units.Zerg.Larva]
                    self.ActionQueue.append(actions.FUNCTIONS.select_point("select", (larva[0].x, larva[0].y)))
                    self.ActionQueue.append(actions.FUNCTIONS.Train_Zergling_quick("now"))
        if action == 5:
            self.ActionQueue.append(actions.FUNCTIONS.select_army("select"))
            self.ActionQueue.append(actions.FUNCTIONS.Attack_screen("now", position))

    def get_game_action(self):
        if len(self.ActionQueue) != 0:
            return self.ActionQueue.popleft()
        else:
            return actions.FUNCTIONS.no_op()