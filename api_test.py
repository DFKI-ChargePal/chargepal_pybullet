import gym
import gym_chargepal


def main() -> None:

    env = gym.make("ChargePal-P2P-1D-PositionControl-v0")
    env.action_space.seed(42)

    obs = env.reset()

    episode = 1
    print(f'Episode: {episode}')
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action=action)

        act_string = " ".join(format(f, '6.2f') for f in action)
        obs_string = " ".join(format(f, '6.2f') for f in obs)

        print(f'Action: {act_string}  ----  Observation: {obs_string}  ----  Reward: {reward:5.2}')

        if done:
            print(f'\n\nEpisode: {episode}')
            episode += 1
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()
