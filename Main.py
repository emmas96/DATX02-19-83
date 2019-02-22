from Agent import Agent
from FrozenLake import FrozenLake

# Constants
EPOCHS = 1000
MOVES = 20


def main():
    game = FrozenLake()
    agent = Agent(game.getNumStates(), game.getNumActions())
    num_wins = 0
    for epoch in range(EPOCHS):
        game.resetGame()
        for move in range(MOVES):
            state = game.getState()
            action = agent.getAction(state)
            next_state, reward, done = game.play(action)
            agent.remember(state, action, reward, next_state, done)
            num_wins += reward
            if done:
                print("epoch: {}/{}, reward: {}".format(epoch, EPOCHS, reward))
                print("number of wins: " + str(num_wins) + ", number of moves: " + str(move+1))
                break
            if agent.getMemoryLength() > agent.getBatchSize():
                agent.train()
    #print("Win rate: " + str((0.0+last_num_wins)/100.0))


if __name__ == "__main__":
    main()