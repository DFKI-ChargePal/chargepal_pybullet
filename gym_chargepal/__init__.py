import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ChargePal-P2P-1D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentP2PCartesian1DPositionCtrl',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)