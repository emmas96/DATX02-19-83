from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from random_agent.SC2TestAgent import SimpleAgent

EPOCHS = 100


def main(unused_argv):

    agent = SimpleAgent()
    points = 0
    try:

        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.zerg),
                         sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True),
                step_mul=8,
                game_steps_per_episode=0,
                visualize=True) as env:

            agent.setup(env.observation_spec(), env.action_spec())
            for epoch in range(EPOCHS):
                agent.reset_game()
                timesteps = env.reset()
                agent.reset()
                if agent.getMemoryLength() > agent.getBatchSize():
                    agent.train()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

                print("epoch: {}/{}, reward: {} Epsilon: {}".format(epoch, EPOCHS, agent.reward, agent.EPSILON))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)