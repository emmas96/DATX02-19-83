from ActionAgent import Agent as aa
from PositionAgent import Agent as pa
from FrozenLake import FrozenLake
from TicTacToe import TicTacToe
import keras
import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Constants
EPOCHS = 200
MOVES = 20


def playFrozenLake():
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
                #print(str(epoch) + "," + str(num_wins))
                break

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
                game.printBoard()
                break

    save_model = input("Save model? (y/n) ")
    if save_model == 'y':
        name = input("Name of model")
        agentX.saveModel(f"TicTacToe-X-{name}")
        agentO.saveModel(f"TicTacToe-O-{name}")

    while True:
        game.resetGame()
        for move in range(MOVES):
            agent = agentList[0]
            state = game.getState()
            action = agent.getAction(state)
            if agent == agentX:
                player = 1
            else:
                player = -1
            next_state, reward, done = game.play(player, action)
            if done:
                game.printBoard()
                print("AI Win")
                break

            game.printBoard()
            action = input("välj var: ")
            next_state, reward, done = game.play(-1, int(action))
            if done:
                game.printBoard()
                print("Player Win")
                break


def main():
    playTicTacToe()
    #playFrozenLake()


if __name__ == "__main__":
    main()


