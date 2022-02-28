import gym


def main(env_name: str) -> None:

    env = gym.make("gym_chargepal:" + env_name)
    env.action_space.seed(42)

    obs = env.reset()

    episode = 1
    print(f'Start test with environment: {env_name}')
    print(f'Episode: {episode}')
    while True:
        env.render()
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action=action)

        act_string = " ".join(format(f, '6.2f') for f in action)
        obs_string = " ".join(format(f, '6.2f') for f in obs)

        print(f'Action: {act_string}  ----  Observation: {obs_string}  ----  Reward: {reward:5.2}')

        if done:
            print(f'\n\nEpisode: {episode}')
            episode += 1
            if episode > 7:
                break
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    env_names = [
        'ChargePal-P2P-1D-PositionControl-v0',
    ]

    for env_ in env_names:
        main(env_)
