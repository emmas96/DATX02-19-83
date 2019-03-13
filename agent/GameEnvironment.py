from pysc2.lib import actions, units, features
import numpy as np
from collections import deque


class GE:

    def __init__(self, x):
        self.Num = x
        self.ActionQueue = deque(maxlen=100)

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
        if len(self.ActionQueue) >= 90:
            return
        if action == 0:
            if len(self.ActionQueue) > 0:
                return
            self.ActionQueue.append((actions.FUNCTIONS.no_op(),
                                     actions.FUNCTIONS.no_op.id))
        if action == 1:
            larva = [unit for unit in obs.observation.feature_units
                     if unit.unit_type == units.Zerg.Larva]
            if len(larva) == 0:
                return
            self.ActionQueue.append((actions.FUNCTIONS.select_point("select", (larva[0].x, larva[0].y)),
                                    actions.FUNCTIONS.select_point.id))
            self.ActionQueue.append((actions.FUNCTIONS.Train_Drone_quick("now"),
                                     actions.FUNCTIONS.Train_Drone_quick.id))
        if action == 2:
            larva = [unit for unit in obs.observation.feature_units
                     if unit.unit_type == units.Zerg.Larva]
            if len(larva) == 0:
                return
            self.ActionQueue.append((actions.FUNCTIONS.select_point("select", (larva[0].x, larva[0].y)),
                                     actions.FUNCTIONS.select_point.id))
            self.ActionQueue.append((actions.FUNCTIONS.Train_Overlord_quick("now"),
                                     actions.FUNCTIONS.Train_Overlord_quick.id))
            self.ActionQueue.append((actions.FUNCTIONS.Rally_Units_minimap("now", (0, 0)),
                                     actions.FUNCTIONS.Rally_Units_minimap.id))
        if action == 5:
            self.ActionQueue.append((actions.FUNCTIONS.select_army("select"),
                                     actions.FUNCTIONS.select_army.id))
            self.ActionQueue.append((actions.FUNCTIONS.Attack_screen("now", position),
                                     actions.FUNCTIONS.Attack_screen.id))

    def get_game_action(self, obs):
        if len(self.ActionQueue) != 0:
            (a, i) = self.ActionQueue.popleft()
            print(a)
            if i in obs.observation.available_actions:
                return a
            else:
                return actions.FUNCTIONS.no_op()
        else:
            return actions.FUNCTIONS.no_op()
