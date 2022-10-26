import gym
import numpy as np

# constants
N_EPISODES = 1


def main(env_name: str, gui: bool) -> None:

    print(3 * '\n' + f"Start test with environment: {env_name}")
    env = gym.make("gym_chargepal:" + env_name)
    env.action_space.seed(42)
    if gui: env.render()
    obs = env.reset()

    episode = 1
    ep_return = 0.0
    print(f'Episode: {episode}')
    while True:

        action = env.action_space.sample()
        # action = np.zeros_like(action)
        obs, reward, done, _ = env.step(action=action)
        ep_return += reward

        act_string = " ".join(format(f, '6.2f') for f in action)
        obs_string = " ".join(format(f, '6.2f') for f in obs)

        print(f'Action: {act_string}  ----  Observation: {obs_string}  ----  Reward: {reward:5.2}')
        if gui: env.render()
        # reset environment
        if done:
            print(f'Finish episode {episode} with return: {ep_return}')
            episode += 1
            ep_return = 0.0
            if episode > N_EPISODES:
                break
            print(f'\n\nEpisode: {episode}')
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    env_names = [
        'ChargePal-Reacher-1D-PositionControl-v0',
        'ChargePal-Reacher-3D-PositionControl-v0',
        'ChargePal-Reacher-6D-PositionControl-v0',
        'ChargePal-Plugger-6D-PositionControl-v0',
        'ChargePal-Plugger-6D-PositionControl-v1',
        'ChargePal-Plugger-6D-PositionControl-v2',
        'ChargePal-Reacher-1D-VelocityControl-v0',
        'ChargePal-Reacher-3D-VelocityControl-v0',
        'ChargePal-Reacher-6D-VelocityControl-v0',
        'ChargePal-Testbed-Plugger-6D-PositionControl-v0',
    ]

    for env_ in env_names:
        main(env_name=env_, gui=True)
