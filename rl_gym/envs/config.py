""" Default configuration and hyper parameters for environment objects. """
import numpy as np
from gym import spaces


ENVIRONMENT = {
    'T': 1000,
}

ENVIRONMENT_PTP_1DOF_CARTESIAN_POSITION_CONTROL = {
    'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(1,), dtype=np.float32),
    'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
}

ENVIRONMENT_PTP_3DOF_CARTESIAN_POSITION_CONTROL = {
    'action_space': spaces.Box(low=-np.sqrt(np.e),  high=np.sqrt(np.e), shape=(3,), dtype=np.float32),
    'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
}

ENVIRONMENT_PTP_3DOF_CARTESIAN_VELOCITY_CONTROL = {
    'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),

    # distance to target(3d), linear velocity(3d)
    'observation_space': spaces.Box(low=-1.0, high=1.0, shape=(3 + 3,), dtype=np.float32),
}


ENVIRONMENT_CONTINUES_PIH_CARTESIAN = {
    'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3+3,), dtype=np.float32),

    # distance to target(3d), rotation quaternion to target(4d), linear velocity(3d), angular velocity(3d)
    'observation_space': spaces.Box(low=-1.0, high=1.0, shape=(3+4+3+3,), dtype=np.float32),
}

ENVIRONMENT_CURRICULUM_PIH = {
    'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3+3,), dtype=np.float32),
    # distance to target(3d), rotation quaternion to target(4d), linear velocity(3d), angular velocity(3d)
    'observation_space': spaces.Box(low=-1.0, high=1.0, shape=(3+4+3+3,), dtype=np.float32),

    # curriculum
    'history_length': 1000,
}
