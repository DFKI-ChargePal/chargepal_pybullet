from __future__ import annotations

# global
import gym
import logging
import argparse
import numpy as np

# local
from gym_chargepal.envs import environment_register

# typing
from typing import Any


LOGGER = logging.getLogger(__name__)


def main(env_name: str, n_episodes: int, gui: bool) -> None:


    LOGGER.info(f"Run with environment: {env_name}")
    # kwargs_cfg = {
    #     'world': {
    #         'gui_txt': 'Hello World'
    #     }
    # }
    kwargs_cfg: dict[str, Any] = {}
    env = gym.make("gym_chargepal:" + env_name, **{'kwargs': kwargs_cfg})
    env.action_space.seed(42)
    if gui: env.render()
    obs = env.reset()

    episode = 1
    run_inf = n_episodes < 0
    ep_return = 0.0
    n_step = 0
    while True:

        action = env.action_space.sample()
        # action = np.zeros_like(action)
        # if n_step <= 67:
        #     action[2] = 1.0
        obs, reward, done, _ = env.step(action=action)
        n_step += 1
        ep_return += reward

        act_string = " ".join(format(f, '6.2f') for f in action)
        obs_string = " ".join(format(f, '6.2f') for f in obs)

        LOGGER.debug(f'Action: {act_string}  ----  Observation: {obs_string}  ----  Reward: {reward:5.2}')
        if gui: env.render()
        # reset environment
        if done:
            LOGGER.info(f'Finish episode {episode} with return: {ep_return}')
            episode += 1
            ep_return = 0.0
            if not run_inf and episode > n_episodes:
                break
            obs = env.reset()
            n_step = 0

    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gym environment test for users')
    parser.add_argument('--env', type=str, default='', help="Option to specify an environment.")
    parser.add_argument('--episodes', type=int, default=1, help='Option to specify the number of episodes. -1 means infinity loop.')
    parser.add_argument('--no_gui', action='store_true', help="Option to omit GUI")
    parser.add_argument('--debug', action='store_true', help="Write debug messages")
    # Parse input arguments
    args = parser.parse_args()
    env_name = args.env
    n_ep = args.episodes
    use_gui = not args.no_gui

    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    LOGGER.info(f"Start API user test")

    if env_name:
        if env_name in environment_register:
            main(env_name, n_ep, use_gui)
        else:
            LOGGER.warn(f"Environment with name '{env_name}' not found. End program...")
    else:
        for env_ in environment_register:
            main(env_, n_ep, use_gui)

    LOGGER.info(f"API user test finished")
