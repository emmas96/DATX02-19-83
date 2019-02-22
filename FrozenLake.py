import numpy as np

BOARD_SIZE = 4
board = [['f', 'f', 'f', 'f'],
         ['f', 'h', 'f', 'h'],
         ['f', 'f', 'f', 'h'],
         ['h', 'f', 'f', 'g']]
# Rewards
REWARD_WIN = 1
REWARD_LOSS = 0
REWARD_OUT = 0
REWARD_FROZEN = 0


class FrozenLake:
    def __init__(self):
        self.posrow = 0
        self.poscol = 0

    # Play a move in the game and receive a reward
    def play(self, action):
        if action == 0:
            self.posrow -= 1
        elif action == 1:
            self.poscol += 1
        elif action == 2:
            self.posrow += 1
        elif action == 3:
            self.poscol -= 1

        if self.posrow < 0 or self.posrow > 3 or self.poscol < 0 or self.poscol > 3:
            reward = REWARD_OUT
            done = True
        elif board[self.posrow][self.poscol] == "h":
            reward = REWARD_LOSS
            done = True
        elif board[self.posrow][self.poscol] == "g":
            reward = REWARD_WIN
            done = True
        else:
            reward = REWARD_FROZEN
            done = False
        if not done:
            next_state = self.getState()
        else:
            next_state = np.zeros([1, BOARD_SIZE ** 2])
        return next_state, reward, done

    # Reset game position to start position
    def resetGame(self):
        self.posrow = 0
        self.poscol = 0

    # Get current state
    def getState(self):
        state = np.zeros([1, BOARD_SIZE ** 2])
        state[0][BOARD_SIZE * self.posrow + self.poscol] = 1
        return state

    def getNumStates(self):
        return BOARD_SIZE ** 2

    def getNumActions(self):
        return BOARD_SIZE

