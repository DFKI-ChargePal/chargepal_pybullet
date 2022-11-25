# global imports
import gym
import numpy as np

# constants
N_EPISODES = 1

# typing
from typing import Any, Dict


def main(env_name: str, gui: bool) -> None:

    print(3 * '\n' + f"Start test with environment: {env_name}")
    # kwargs_cfg = {
    #     'world': {
    #         'gui_txt': 'Hello World'
    #     }
    # }
    kwargs_cfg: Dict[str, Any] = {}
    env = gym.make("gym_chargepal:" + env_name, **{'kwargs': kwargs_cfg})
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

    # test for all registered environments
    from gym_chargepal.envs import environment_register

    # main(env_name="ChargePal-Testbed-Plugger-PositionControl-v0", gui=True)
    for env_ in environment_register:
        main(env_name=env_, gui=True)
