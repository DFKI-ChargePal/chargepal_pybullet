import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ChargePal-P2P-1D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentTcpPositionCtrlPtP1Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-P2P-3D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentTcpPositionCtrlPtP3Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-P2P-6D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentTcpPositionCtrlPtP6Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-PiH-6D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentTcpPositionCtrlPiH6Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-PiH-6D-PositionControl-v1',
    entry_point='gym_chargepal.envs:EnvironmentTcpPositionCtrlPiH6DofV1',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-PiH-6D-PositionControl-v2',
    entry_point='gym_chargepal.envs:EnvironmentTcpPositionCtrlPiH6DofV2',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-P2P-1D-VelocityControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentTcpVelocityCtrlPtP1Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-P2P-3D-VelocityControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentTcpVelocityCtrlPtP3Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-P2P-6D-VelocityControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentTcpVelocityCtrlPtP6Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)