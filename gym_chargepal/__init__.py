# global
import logging
import gymnasium.envs.registration as gym_registration

# local
from gym_chargepal.envs import environment_register


LOGGER = logging.getLogger(__name__)


for env_id, env_kw in environment_register.items():
    gym_registration.register(
        id=env_id,
        entry_point='gym_chargepal.envs.factory:EnvironmentFactory',
        reward_threshold=1.0,
        nondeterministic=True,
        kwargs=env_kw,
    )
