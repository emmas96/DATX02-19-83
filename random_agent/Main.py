from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from random_agent.SC2TestAgent import SimpleAgent
import tensorflow as tf
import time
EPOCHS = 2000


def main(unused_argv):
    i = 0
    agent = SimpleAgent()
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
                step_mul=64,
                game_steps_per_episode=0,
                visualize=True) as env:

            agent.setup(env.observation_spec(), env.action_spec())
            for epoch in range(EPOCHS):
                agent.reset_game()
                timesteps = env.reset()
                agent.reset()

                print(str(agent.getMemoryLength()))
                if agent.getMemoryLength() > agent.getBatchSize():
                    agent.train()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
                file = open("plot.txt", "a")
                file.write(str(epoch + 1) + " , ")
                file.write(str(agent.reward) + " , ")
                file.write(str(timesteps[0].observation['score_cumulative'][0]) + "\n")
                file.close()
                print("epoch: {}/{}, reward: {} Epsilon: {}".format(epoch, EPOCHS, agent.reward, agent.EPSILON))
                for state, action, reward, next_state, done in agent.tmpMemory:
                    if(i < agent.reward):
                        agent.memory.append(
                            (state, action, 1, next_state, False))
                    else:
                        agent.memory.append(
                            (state, action, -1, next_state, False))
                i = agent.reward


            agent.model.save(f"model-test-{time.time()}.h5")

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)