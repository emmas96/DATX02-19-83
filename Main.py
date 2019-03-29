from ActionAgent import Agent as aa
from PositionAgent import Agent as pa
from FrozenLake import FrozenLake
from TicTacToe import TicTacToe
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Constants
EPOCHS = 200
MOVES = 20

##clas
class live_graph():

    def __init__(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        # Wins
        color = 'tab:red'
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        # Epsilon
        color = 'tab:blue'
        ax2.tick_params(axis='y', labelcolor=color)

        self.ax1 = ax1
        self.ax2 = ax2

        self.wins = []
        self.epsilons = []
        self.epochs = []

    def update_graph(self, epoch, number_of_wins, epsilon):
        self.epochs.append(epoch)
        self.wins.append(number_of_wins/(epoch+1))
        self.epsilons.append(epsilon)

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.set_xlabel('epochs')

        color = 'tab:red'
        self.ax1.set_ylabel('nr wins', color=color)
        self.ax1.plot(self.epochs, self.wins, color=color)

        color = 'tab:blue'
        self.ax2.set_ylabel('epsilon', color=color)
        self.ax2.plot(self.epochs, self.epsilons, color=color)

        plt.pause(0.005)

    def keep_show(self):
        plt.show()

def playFrozenLake():
    graph = live_graph()
    game = FrozenLake()
    agent = aa(game.getNumStates(), game.getNumActions())
    num_wins = 0
    for epoch in range(EPOCHS):
        game.resetGame()
        for move in range(MOVES):
            state = game.getState()
            action = agent.getAction(state)
            next_state, reward, done = game.play(action)
            agent.remember(state, action, reward, next_state, done)
            num_wins += reward
            if agent.getMemoryLength() > agent.getBatchSize():
                agent.train()
            if done:
                print("epoch: {}/{}, reward: {}".format(epoch, EPOCHS, reward))
                print("number of wins: " + str(num_wins) + ", number of moves: " + str(move + 1))

                graph.update_graph(epoch, num_wins, agent.get_epsilon())
                # print(str(epoch) + "," + str(num_wins))
                break

    graph.keep_show()
    # print("Win rate: " + str((0.0+last_num_wins)/100.0))


def playTicTacToe():
    game = TicTacToe()
    agentX = pa(game.getNumStates(), game.getNumActions())
    agentO = pa(game.getNumStates(), game.getNumActions())
    agentList = [agentX, agentO]
    num_wins = [0, 0]
    for epoch in range(EPOCHS):
        game.resetGame()
        for move in range(MOVES):
            if move > 9:
                print("Wrong")
            agent = agentList[move % 2]
            state = game.getState()
            action = agent.getAction(state)
            if agent == agentX:
                player = 1
            else:
                player = -1
            next_state, reward, done = game.play(player, action)
            agent.remember(state, action, reward, next_state, done)
            num_wins[move % 2] += reward
            if agent.getMemoryLength() > agent.getBatchSize():
                agent.train()
            if done:
                print("epoch: {}/{}, reward: {}".format(epoch, EPOCHS, reward))
                print("number of X wins: " + str(num_wins[0]) +
                      ", number of O wins: " + str(num_wins[1]) +
                      ", number of moves: " + str(move + 1))
                print(game.getBoard())
                break
    while True:
        game.resetGame()
        for move in range(MOVES):
            print(state)
            if move > 9:
                print("Wrong")
            agent = agentList[0]
            state = game.getState()
            action = agent.getAction(state)
            if agent == agentX:
                player = 1
            else:
                player = -1
            next_state, reward, done = game.play(player, action)
            if done:
                print(game.getState())
                print("AI Win")
                break
            state = game.getState()
            print(state)
            action = input("Choose where: ")
            next_state, reward, done = game.play(-1, int(action))
            if done:
                print(game.getState())
                print("Player Win")
                break





def main():
    # playTicTacToe()
    playFrozenLake()



if __name__ == "__main__":
    main()



