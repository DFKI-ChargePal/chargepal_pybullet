import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ChargePal-Reacher-1D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentReacherPositionCtrl1Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Reacher-3D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentReacherPositionCtrl3Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Reacher-6D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentReacherPositionCtrl6Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Plugger-6D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentPluggerPositionCtrl6DofV0',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Plugger-6D-PositionControl-v1',
    entry_point='gym_chargepal.envs:EnvironmentPluggerPositionCtrl6DofV1',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Plugger-6D-PositionControl-v2',
    entry_point='gym_chargepal.envs:EnvironmentPluggerPositionCtrl6DofV2',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Testbed-Plugger-6D-PositionControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentTestbedPluggerPositionCtrl6DofV0',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Reacher-1D-VelocityControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentReacherVelocityCtrl1Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Reacher-3D-VelocityControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentReacherVelocityCtrl3Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='ChargePal-Reacher-6D-VelocityControl-v0',
    entry_point='gym_chargepal.envs:EnvironmentReacherVelocityCtrl6Dof',
    reward_threshold=1.0,
    nondeterministic=True,
)