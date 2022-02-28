# test joint/link name to index mapping

import pybullet as p
import pybullet_data


from gym_chargepal.bullet.utility import (
    create_joint_index_dict,
    create_link_index_dict,
)


class TestBulletUtility:

    panda_joint_names = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
        'panda_joint8',
    ]

    panda_link_names = [
        'panda_link1',
        'panda_link2',
        'panda_link3',
        'panda_link4',
        'panda_link5',
        'panda_link6',
        'panda_link7',
        'panda_link8',
    ]

    index_list = [_ for _ in range(8)]

    def start_sim(self) -> None:
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    def end_sim(self) -> None:
        p.disconnect()

    def test_create_link_index(self) -> None:
        self.start_sim()
        link_idx_dict = create_link_index_dict(self.panda_id, self.panda_link_names, self.physics_client)
        self.end_sim()

        for name, index in link_idx_dict.items():
            assert name in self.panda_link_names
            assert index in self.index_list
        
    def test_create_joint_index(self) -> None:
        self.start_sim()
        joint_idx_dict = create_joint_index_dict(self.panda_id, self.panda_joint_names, self.physics_client)
        self.end_sim()

        for name, index in joint_idx_dict.items():
            assert name in self.panda_joint_names
            assert index in self.index_list
