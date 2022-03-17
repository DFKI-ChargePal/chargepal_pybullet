""" This file defines the default configuration of the pybullet worlds. """
import numpy as np
import pybullet as p


ur_joint_names = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


ur_joint_start_config = {
    'shoulder_pan_joint': 0.0,
    'shoulder_lift_joint': -np.pi/2 - np.pi/6,
    'elbow_joint': -np.pi/2 - np.pi/6,
    'wrist_1_joint': np.pi/2 - np.pi/6,
    'wrist_2_joint': np.pi/2,
    'wrist_3_joint': np.pi/2,
}


ur_link_names = [
    'shoulder_link',
    'upper_arm_link',
    'forearm_link',
    'wrist_1_link',
    'wrist_2_link',
    'wrist_3_link',
]

plug_ref_frame_names = [
    'virt_tool_frame_x',
    'virt_tool_frame_y',
    'virt_tool_frame_z',
]

adpstd_ref_frame_names = [
    'virt_tgt_frame_x',
    'virt_tgt_frame_y',
    'virt_tgt_frame_z',
]

WORLD = {
    'hz_sim': 240,  # frequency physic engine !not recommended to changing this value!
    'hz_ctrl': 60,
    'gravity': (0, 0, -9.81),
    'ur_joint_names': ur_joint_names,
    'ur_joint_start_config': ur_joint_start_config,
    'chargepal_description_pkg': 'chargepal_description',
    'urdf_sub_dir': '_bullet_urdf_autogen',
}

WORLD_PTP = {
    'robot_urdf': 'primitive_chargepal_with_fix_plug.urdf',
    'robot_start_pos': [0.0, 1.15, 0.0],
    'robot_start_ori': p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
    'plug_reference_frame': 'plug_reference_frame',
    'plug_ref_frame_names': plug_ref_frame_names,
    'target_pos': None,
    'target_ori': None,
}

WORLD_PIH = {
    'robot_urdf': 'primitive_chargepal_with_fix_plug.urdf',
    'adapter_station_urdf': 'primitive_adapter_station.urdf',
    'ft_sensor_joint': 'measurement_joint',
    'plug_reference_frame': 'plug_reference_frame',
    'adpstd_reference_frame': 'target_reference_frame',
    'plug_ref_frame_names': plug_ref_frame_names,
    'adpstd_ref_frame_names': adpstd_ref_frame_names,
}
