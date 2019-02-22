from ActionAgent import Agent as aa
from PositionAgent import Agent as pa
from FrozenLake import FrozenLake
from TicTacToe import TicTacToe

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
            if move > 9:
                print("FEL")
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
                print("FEL")
            agent = agentList[0]
            state = game.getState()
            action = agent.getAction(state)
            if agent == agentX:
                player = 1
            else:
                player = -1
            next_state, reward, done = game.play(player, action)
            if(done):
                print(game.getState())
                print("AI Win")
                break;
            state = game.getState()
            print(state)
            action = input("välj var")
            next_state, reward, done  = game.play(-1, int(action))
            if (done):
                print(game.getState())
                print("Player Win")
                break;



def main():
    playTicTacToe()



if __name__ == "__main__":
    main()