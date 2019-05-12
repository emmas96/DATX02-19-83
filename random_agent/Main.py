from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from random_agent.SC2TestAgent import SimpleAgent
import tensorflow as tf
import time
EPOCHS = 300


def main(unused_argv):
    for gamma in [0.9]:
        for Mb in [512]:
            for Et in [0.2]:
                for index in [0]:
                    i = 0
                    agent = SimpleAgent()
                    agent.imi = 0
                    agent.GAMMA = gamma
                    agent.BATCH_SIZE = Mb
                    agent.EPSILON_TO = Et
                    agent.EPSILON = 1

                    #agent.model = tf.keras.models.load_model("model-test-1554279092.6362414.h5")
                    points = 0
                    try:

                        with sc2_env.SC2Env(
                                map_name="Simple64",
                                players=[sc2_env.Agent(sc2_env.Race.zerg),
                                         sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                                    use_feature_units=True),
                                step_mul=22,
                                game_steps_per_episode=0,
                                visualize=True) as env:

                            agent.setup(env.observation_spec(), env.action_spec())




                            agent.setup(env.observation_spec(), env.action_spec())
                            for epoch in range(EPOCHS):
                                agent.reset_game()
                                agent.epoch = epoch
                                agent.c = epoch
                                timesteps = env.reset()
                                agent.reset()
                                if agent.getMemoryLength() > agent.getBatchSize():
                                    agent.train()
                                while True:
                                    step_actions = [agent.step(timesteps[0])]
                                    if timesteps[0].last():
                                        break
                                    timesteps = env.step(step_actions)
                                # agent.save_plot_data(agent.reward / (epoch + 1))
                                # file.write("hej")
                                for state, action, reward, next_state, done in agent.tmpMemory:
                                    if (i > agent.reward):
                                        agent.memory.append(
                                            (state, action, reward, next_state, False))
                                    else:
                                        agent.memory.append(
                                            (state, action, reward, next_state, False))
                                i = agent.reward
                                file = open(f"Data/GAME/plot_Train_addR_GAME_G{gamma}_Et{Et}_Mb{Mb}_imi{agent.imi}_I_{index}.txt",
                                            "a")
                                file.write(str(epoch) + ", ")
                                file.write(str(agent.reward) + ", ")
                                file.write(str(timesteps[0].observation['score_cumulative'][0]) + ", ")
                                file.write(str(agent.reward / (epoch + 1)) + ", ")
                                file.write(str(agent.EPSILON) + "\n")
                                file.close()
                                print(str(agent.getMemoryLength()))
                                # ani = animation.FuncAnimation(fig, plotdata, interval=1000)
                                # plt.show()
                                print("epoch: {}/{}, reward: {} Epsilon: {}".format(epoch, 1000, agent.reward,
                                                                                    agent.EPSILON))
                            # agent.model.save(f"model-test-{time.time()}.h5")
                            # Validate
                            for epoch in range(10):
                                agent.reset_game()
                                agent.BATCH_SIZE = Mb
                                agent.EPSILON = Et
                                agent.EPSILON_TO = Et
                                agent.GAMMA = gamma
                                agent.reward = 0
                                timesteps = env.reset()
                                agent.reset()

                                while True:
                                    step_actions = [agent.step(timesteps[0])]
                                    if timesteps[0].last():
                                        break
                                    timesteps = env.step(step_actions)
                                # agent.save_plot_data(agent.reward / (epoch + 1))
                                # file.write("hej")
                                file = open(f"Data/Game/plot_Valid_addR_GAME_G{gamma}_Et{Et}_Mb{Mb}_imi{agent.imi}_I_{index}.txt","a")
                                file.write(str(epoch) + ", ")
                                file.write(str(agent.reward) + ", ")
                                file.write(str(timesteps[0].observation['score_cumulative'][0]) + ", ")
                                file.write(str(agent.reward / (epoch + 1)) + ", ")
                                file.write(str(agent.EPSILON) + "\n")
                                file.close()
                                # ani = animation.FuncAnimation(fig, plotdata, interval=1000)
                                # plt.show()
                                print("VALID: epoch: {}/{}, reward: {} Epsilon: {}".format(epoch, EPOCHS, agent.reward,
                                                                                           agent.EPSILON))

                    except KeyboardInterrupt:
                        pass


if __name__ == "__main__":
    app.run(main)
