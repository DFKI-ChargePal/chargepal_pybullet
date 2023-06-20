import gym_chargepal.envs.env_config as config


environment_register = {
    'ChargePal-Testbed-Reacher-3D-PositionControl-v1': config.testbed_reacher_3dof_position_ctrl_v1,
    'ChargePal-Testbed-Reacher-6D-PositionControl-v1': config.testbed_reacher_6dof_position_ctrl_v1,
    'ChargePal-Testbed-Plugger-6D-ForceControl-v1': config.testbed_plugger_6dof_force_ctrl_v1,
    'ChargePal-Testbed-Plugger-6D-MotionControl-v1': config.testbed_plugger_6dof_motion_ctrl_v1,
    'ChargePal-Testbed-Plugger-6D-PositionControl-v1': config.testbed_plugger_6dof_position_ctrl_v1,
    'ChargePal-Testbed-Plugger-6D-ComplianceControl-v1': config.testbed_plugger_6dof_compliance_ctrl_v1,
}
