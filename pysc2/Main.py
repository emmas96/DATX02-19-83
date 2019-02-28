from SC2Test import SC2Test
from SC2TestAgent import SimpleAgent
from pysc2.agents import base_agent
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units



# constants
EPOCHS = 100


def main(unused_argv):
    agent = SimpleAgent()
    points = 0
    try:

        with sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=15, minimap=10),
                    use_feature_units=True),
                step_mul=64,
                game_steps_per_episode=0,
                visualize=True) as env:

            agent.setup(env.observation_spec(), env.action_spec())
            for epoch in range(EPOCHS):
                agent.reset_game()
                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
                    if agent.getMemoryLength() > agent.getBatchSize():
                        agent.train()
                print("epoch: {}/{}, reward: {}".format(epoch, EPOCHS, agent.reward))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)






