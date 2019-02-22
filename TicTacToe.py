import numpy as np

BOARD_SIZE = 3

# Rewards
REWARD_WIN = 1
REWARD_LOSS = 0
REWARD_ACTION = 0

class TicTacToe:
    def __init__(self):
        self.board = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]

    # Play a move in the game and receive a reward
    def play(self, player, action):
        self.board[action/3][action % 3] = player
        reward = REWARD_ACTION
        done = False
        # Check win
        for i in range(3):
            if self.board[i][0] == player and self.board[i][1] == player and self.board[i][2] == player\
                    or self.board[0][i] == player and self.board[1][i] == player and self.board[2][i] == player:
                reward = REWARD_WIN
                done = True
        if self.board[0][0] == player and self.board[1][1] == player and self.board[2][2] == player\
                or self.board[0][2] == player and self.board[1][1] == player and self.board[2][0] == player:
            reward = REWARD_WIN
            done = True
        if done:
            print("player: " + str(player) + " Win")
            return self.getState(), reward, done


        if np.count_nonzero(np.reshape(self.board, [1, self.getNumStates()])) == 9:
            done = True
            print("Lika")
        return self.getState(), reward, done



    def switchPlayer(self):
        self.board = -1 * self.board

    # Reset game position to start position
    def resetGame(self):
        self.board = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]

    def getBoard(self):
        return self.board

    # Get current state
    def getState(self):
        state = np.reshape(self.board, [1, self.getNumStates()])
        return state

    def getNumStates(self):
        return BOARD_SIZE ** 2

    def getNumActions(self):
        return BOARD_SIZE ** 2