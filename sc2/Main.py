from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from sc2.SC2TestAgent import SimpleAgent

EPOCHS = 1000


def main(unused_argv):
    print("Jag lovar du är en duktig agent")
    agent = SimpleAgent()
    lol = agent.model.get_weights()
    x = agent.model.get_weights()
    points = 0
    try:
        with sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=40, minimap=10),
                    use_feature_units=True),
                step_mul=160,
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
                        for state, action, next_state, done in agent.tmpmemory:
                            if next_state is not None and state is not None:
                                bla = timesteps[0].observation['score_cumulative'][0]
                                # TEMP VÄRDE 0.2
                                bla *= 0.2
                                if bla > 0:
                                    agent.memory.append((state, action, bla, next_state, done))
                        agent.tmpmemory.clear()
                        break
                    timesteps = env.step(step_actions)
                print("epoch: {}/{}, reward: {} Epsilon: {}".format(epoch, EPOCHS, agent.reward, agent.EPSILON))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)






