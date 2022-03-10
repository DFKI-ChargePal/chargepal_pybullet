import gym

# constants
N_EPISODES = 2


def main(env_name: str) -> None:

    env = gym.make("gym_chargepal:" + env_name)
    env.action_space.seed(42)
    env.render()
    obs = env.reset()

    episode = 1
    ep_return = 0.0
    print(f'Start test with environment: {env_name}')
    print(f'Episode: {episode}')
    while True:

        action = env.action_space.sample()
        # action = np.zeros_like(action)
        obs, reward, done, _ = env.step(action=action)
        ep_return += reward

        act_string = " ".join(format(f, '6.2f') for f in action)
        obs_string = " ".join(format(f, '6.2f') for f in obs)

        print(f'Action: {act_string}  ----  Observation: {obs_string}  ----  Reward: {reward:5.2}')
        env.render()
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
        'ChargePal-P2P-1D-PositionControl-v0',
        'ChargePal-P2P-3D-PositionControl-v0',
        'ChargePal-P2P-6D-PositionControl-v0',
    ]

    for env_ in env_names:
        main(env_)
