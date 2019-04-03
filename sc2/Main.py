from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from sc2.SC2TestAgent import SimpleAgent
from sc2.SC2ConvAgent import ConvAgent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process
import time

EPOCHS = 20000
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)



def main(unused_argv):
    print("Jag lovar du är en duktig agent")
    agent = SimpleAgent()
    #agent = ConvAgent()
    lol = agent.model.get_weights()
    x = agent.model.get_weights()
    points = 0
    file = open("plot20000epochsAlpha0.0005.txt", "w")
    file.close()
    try:
        with sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=40, minimap=10),
                    use_feature_units=True),
                step_mul=220,
                game_steps_per_episode=0,
                visualize=True) as env:

            agent.setup(env.observation_spec(), env.action_spec())
            for epoch in range(EPOCHS):
                agent.reset_game()
                timesteps = env.reset()
                agent.reset()
                if agent.getMemoryLength() > agent.getBatchSize():
                    agent.train()
                    print(str(agent.getMemoryLength()))
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        for state, action, reward, next_state, done in agent.tmpmemory:
                            if next_state is not None and state is not None:
                                bla = timesteps[0].observation['score_cumulative'][0]
                                # TEMP VÄRDE 0.2
                                bla *= 0.2
                                if bla > 0:
                                    agent.memory.append((state, action, reward, next_state, done))
                        agent.tmpmemory.clear()
                        print("Egna action: " + str(agent.oa))
                        break
                    timesteps = env.step(step_actions)
                # agent.save_plot_data(agent.reward / (epoch + 1))
                #file.write("hej")
                file = open("plot.txt", "a")
                file.write(str(epoch) + ", ")
                file.write(str(agent.reward) + ", ")
                file.write(str(timesteps[0].observation['score_cumulative'][0]) + ", ")
                file.write(str(agent.reward / (epoch + 1)) + "\n")
                file.close()
                #ani = animation.FuncAnimation(fig, plotdata, interval=1000)
                #plt.show()
                print("epoch: {}/{}, reward: {} Epsilon: {}".format(epoch, EPOCHS, agent.reward, agent.EPSILON))
            agent.model.save(f"model-test-{time.time()}.h5")


    except KeyboardInterrupt:
        pass


def plotdata(i):
    graph_data = file.read()
    lines = graph_data.split("\n")
    ax1.clear()
    ax1.plot(lines)


if __name__ == "__main__":
    app.run(main)






