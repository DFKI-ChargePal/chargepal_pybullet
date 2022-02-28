import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ChargePal-P2P-1D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentP2PCartesian1DPositionCtrl',
    kwargs={
        'config_env': {
            'gui': True,
            'T': 100,
        },
        'config_world': {
            'hz_ctrl': 20,
        },
    },
    reward_threshold=1.0,
    nondeterministic=True,
)