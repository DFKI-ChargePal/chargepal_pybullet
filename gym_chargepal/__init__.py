import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ChargePal-P2P-1D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentPtPCartesianPositionCtrl1DOF',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-P2P-3D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentPtPCartesianPositionCtrl3DOF',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-P2P-6D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentPtPCartesianPositionCtrl6DOF',
    reward_threshold=1.0,
    nondeterministic=True,
)