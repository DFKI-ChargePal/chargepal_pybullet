import gym_chargepal.envs.env_config as config


environment_register = {
    'ChargePal-Testbed-Plugger-PositionControl-v0': config.testbed_plugger_position_ctrl_v0,

    'ChargePal-Plugger-PositionControl-v0': config.plugger_position_ctrl_v0,
    'ChargePal-Plugger-PositionControl-v1': config.plugger_position_ctrl_v1,
    'ChargePal-Plugger-PositionControl-v2': config.plugger_position_ctrl_v2,

    'ChargePal-Reacher-1D-PositionControl-v0': config.reacher_1dof_position_ctrl_v0,
    'ChargePal-Reacher-3D-PositionControl-v0': config.reacher_3dof_position_ctrl_v0,
    'ChargePal-Reacher-6D-PositionControl-v0': config.reacher_6dof_position_ctrl_v0,

    'ChargePal-Reacher-3D-VelocityControl-v0': config.reacher_3dof_velocity_ctrl_v0,
    'ChargePal-Reacher-6D-VelocityControl-v0': config.reacher_6dof_velocity_ctrl_v0,
}
