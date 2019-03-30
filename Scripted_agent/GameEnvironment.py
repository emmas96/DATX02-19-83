from pysc2.lib import actions, units, features
import numpy as np
from collections import deque
import random


class GE:
    def __init__(self, x):
        self.Num = x
        self.ActionQueue = deque(maxlen=8)
        self.enemyPos = None
        self.ourPos = None
        self.overlordPlace = None

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

    def set_game_action(self, action, obs):
        print("yeah boi")
        if len(self.ActionQueue) >= 1:
            return
        if action == 0:
            if len(self.ActionQueue) > 0:
                return
            self.ActionQueue.append((None, actions.FUNCTIONS.no_op(),
                                     actions.FUNCTIONS.no_op.id))
        if action == 1:
            larva = [unit for unit in obs.observation.feature_units
                     if unit.unit_type == units.Zerg.Larva]
            hatchery = [unit for unit in obs.observation.feature_units
                     if unit.unit_type == units.Zerg.Hatchery]
            if len(larva) == 0 or len(hatchery) == 0:
                return
            self.ActionQueue.append((None, actions.FUNCTIONS.select_point("select", (hatchery[0].x, hatchery[0].y)),
                                     actions.FUNCTIONS.select_point.id))
            self.ActionQueue.append((units.Zerg.Hatchery, actions.FUNCTIONS.select_larva(),
                                     actions.FUNCTIONS.select_larva.id))
            self.ActionQueue.append((units.Zerg.Larva, actions.FUNCTIONS.Train_Drone_quick("now"),
                                     actions.FUNCTIONS.Train_Drone_quick.id))
        if action == 2:
            larva = [unit for unit in obs.observation.feature_units
                     if unit.unit_type == units.Zerg.Larva]
            hatchery = [unit for unit in obs.observation.feature_units
                   if unit.unit_type == units.Zerg.Hatchery]
            if len(larva) == 0 or len(hatchery) == 0:
                return
            self.ActionQueue.append((None, actions.FUNCTIONS.select_point("select", (hatchery[0].x, hatchery[0].y)),
                                     actions.FUNCTIONS.select_point.id))
            self.ActionQueue.append((units.Zerg.Hatchery, actions.FUNCTIONS.select_larva(),
                                     actions.FUNCTIONS.select_larva.id))
            self.ActionQueue.append((units.Zerg.Larva, actions.FUNCTIONS.Train_Overlord_quick("now"),
                                     actions.FUNCTIONS.Train_Overlord_quick.id))
            self.ActionQueue.append((None, actions.FUNCTIONS.Rally_Units_minimap("now", self.overlordPlace),
                                     actions.FUNCTIONS.Rally_Units_minimap.id))
        if action == 3:

            drones = [unit for unit in obs.observation.feature_units
                      if unit.unit_type == units.Zerg.Drone]

            if len(drones) > 0:
                drone = random.choice(drones)
                if drone.x < 0 or drone.x > 83 or drone.y < 0 or drone.y > 83:
                    return
                self.ActionQueue.append((None, actions.FUNCTIONS.select_point("select", (drone.x, drone.y)),
                                         actions.FUNCTIONS.select_point.id))
                x = random.randint(0, 83)
                y = random.randint(0, 83)
                self.ActionQueue.append((units.Zerg.Drone, actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y)),
                                         actions.FUNCTIONS.Build_SpawningPool_screen.id))
        if action == 4:
            larva = [unit for unit in obs.observation.feature_units
                     if unit.unit_type == units.Zerg.Larva]
            hatchery = [unit for unit in obs.observation.feature_units
                        if unit.unit_type == units.Zerg.Hatchery]
            if len(larva) == 0 or len(hatchery) == 0:
                return
            self.ActionQueue.append((None, actions.FUNCTIONS.select_point("select", (hatchery[0].x, hatchery[0].y)),
                                     actions.FUNCTIONS.select_point.id))
            self.ActionQueue.append((units.Zerg.Hatchery, actions.FUNCTIONS.select_larva(),
                                     actions.FUNCTIONS.select_larva.id))
            self.ActionQueue.append((units.Zerg.Larva, actions.FUNCTIONS.Train_Zergling_quick("now"),
                                     actions.FUNCTIONS.Train_Zergling_quick.id))
            self.ActionQueue.append((None, actions.FUNCTIONS.Rally_Units_minimap("now", self.ourPos),
                                     actions.FUNCTIONS.Rally_Units_minimap.id))
        if action == 5:
            self.ActionQueue.append((None, actions.FUNCTIONS.select_army("select"),
                                     actions.FUNCTIONS.select_army.id))
            self.ActionQueue.append((units.Zerg.Zergling, actions.FUNCTIONS.Attack_minimap("now", self.enemyPos),
                                     actions.FUNCTIONS.Attack_minimap.id))

    def get_game_action(self, obs):
        if len(self.ActionQueue) != 0:
            (t, a, i) = self.ActionQueue.popleft()
            print(a)
            if i in obs.observation.available_actions:
                if t is not None:
                    if self.unit_type_is_selected(obs, t):
                        return a
                    else:
                        return actions.FUNCTIONS.no_op()
                else:
                    return a
            else:
                return actions.FUNCTIONS.no_op()
        else:
            return actions.FUNCTIONS.no_op()

    def _xy_locs(self, mask):
        y, x = mask.nonzero()
        return list(zip(x, y))

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False
