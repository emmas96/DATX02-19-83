import numpy as np


class FrozenLake:
    def __init__(self):
        self.BOARD_SIZE = 4
        self.board = [['f', 'f', 'f', 'f'],
                      ['f', 'h', 'f', 'h'],
                      ['f', 'f', 'f', 'h'],
                      ['h', 'f', 'f', 'g']]
        self.posrow = 0
        self.poscol = 0

        # Rewards
        self.REWARD_WIN = 1
        self.REWARD_LOSS = 0
        self.REWARD_OUT = 0
        self.REWARD_FROZEN = 0

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
            reward = self.REWARD_OUT
            done = True
        elif self.board[self.posrow][self.poscol] == "h":
            reward = self.REWARD_LOSS
            done = True
        elif self.board[self.posrow][self.poscol] == "g":
            reward = self.REWARD_WIN
            done = True
        else:
            reward = self.REWARD_FROZEN
            done = False
        if not done:
            next_state = self.getState()
        else:
            next_state = np.zeros([1, self.BOARD_SIZE ** 2])
        return next_state, reward, done

    # Reset game position to start position
    def resetGame(self):
        self.posrow = 0
        self.poscol = 0

    # Get current state
    def getState(self):
        state = np.zeros([1, self.BOARD_SIZE ** 2])
        state[0][self.BOARD_SIZE * self.posrow + self.poscol] = 1
        return state

    def getNumStates(self):
        return self.BOARD_SIZE ** 2

    def getNumActions(self):
        return self.BOARD_SIZE

