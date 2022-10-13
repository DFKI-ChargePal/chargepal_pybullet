import logging
import pybullet as p


from gym_chargepal.bullet.config import (
    BulletJointInfo,
)


# mypy
from pybullet_utils.bullet_client import BulletClient
from typing import (
    Tuple, 
    List,
    Dict,
)


LOGGER = logging.getLogger(__name__)


def draw_sphere_marker(
        position: Tuple[float, ...],
        radius: float, 
        color: Tuple[float, ...],
        bullet_client: BulletClient
    ) -> int:

    """ Function to draw a sphere marker """
    vs_id = bullet_client.createVisualShape(
        p.GEOM_SPHERE, 
        radius=radius, 
        rgbaColor=color
        )
    marker_id: int = bullet_client.createMultiBody(
        basePosition=position,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vs_id
        )
    return marker_id


def draw_cylinder_marker(
        position: Tuple[float, ...],
        orientation: Tuple[float, ...],
        radius: float, 
        height: float,
        color: Tuple[float, ...],
        bullet_client: BulletClient 
    ) -> int:

    """ Function to draw a cylinder marker """
    vs_id = bullet_client.createVisualShape(
        p.GEOM_CYLINDER, 
        radius, 
        [1.0, 1.0, 1.0],
        height,  
        rgbaColor=color,
        visualFramePosition=(0.0, 0.0, height/2)
        )
    marker_id: int = bullet_client.createMultiBody(
        basePosition=position,
        baseOrientation=orientation,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vs_id
    )
    return marker_id


def _extract_name(byte_name: bytes) -> str:
    """ Helper function to extract the name from a byte string. """
    return list(filter(None, str(byte_name).split("'")))[-1]


def get_link_idx(body_id: int, link_name: str, bullet_client: BulletClient) -> int:
    """ Utility function to extract link index.

    :param body_id:   The bullet body unique id
    :param link_name: The URDF link name
    :param client_id: Bullet physical client id
    :return:          Link name corresponding index. In case of link not found -2.
    """
    link_idx = -2
    for idx in range(bullet_client.getNumJoints(bodyUniqueId=body_id)):
        joint_info = bullet_client.getJointInfo(bodyUniqueId=body_id, jointIndex=idx)
        scanned_link_name = _extract_name(joint_info[BulletJointInfo.LINK_NAME])
        scanned_link_idx = joint_info[BulletJointInfo.JOINT_INDEX]
        if scanned_link_name == link_name:
            link_idx = scanned_link_idx
            break
    return link_idx


def get_joint_idx(body_id: int, joint_name: str, bullet_client: BulletClient) -> int:
    """ Utility function to extract joint index.
    
    :param body_id:    The bullet body unique id
    :param joint_name: The URDF joint name
    :param client_id:  Bullet physical client id
    :return:           Joint name corresponding index. In case of joint not found -1.
    """
    joint_idx = -2
    for idx in range(bullet_client.getNumJoints(bodyUniqueId=body_id)):
        joint_info = bullet_client.getJointInfo(bodyUniqueId=body_id, jointIndex=idx)
        scanned_joint_name = _extract_name(joint_info[BulletJointInfo.JOINT_NAME])
        scanned_joint_idx = joint_info[BulletJointInfo.JOINT_INDEX]
        if scanned_joint_name == joint_name:
            joint_idx = scanned_joint_idx
            break
    return joint_idx


def create_link_index_dict(body_id: int, link_names: List[str], bullet_client: BulletClient) -> Dict[str, int]:
    """ Utility function to map a list of link names to the corresponding indices in the physical model.
    
    :param body_id:    The bullet body unique id
    :param joint_name: List of URDF link names
    :param client_id:  Bullet physical client id
    :return:           Dictionary with link name/index mapping. Key: link name - Value: link index
    """
    link_index_dict = {}
    for link_name in link_names:
        link_idx = get_link_idx(body_id=body_id, link_name=link_name, bullet_client=bullet_client)
        if link_idx < -1:
            LOGGER.warning(f"A link with the name '{link_name}' was not found")
        else:
            link_index_dict[link_name] = link_idx

    return link_index_dict


def create_joint_index_dict(body_id: int, joint_names: List[str], bullet_client: BulletClient) -> Dict[str, int]:
    """ Utility function to map a list of joint names to the corresponding indices in the physical model.
    
    :param body_id:    The bullet body unique id
    :param joint_name: List of URDF joint names
    :param client_id:  Bullet physical client id
    :return:           Dictionary with joint name/index mapping. Key: joint name - Value: joint index
    """
    joint_index_dict = {}
    for joint_name in joint_names:
        joint_idx = get_joint_idx(body_id=body_id, joint_name=joint_name, bullet_client=bullet_client)
        if joint_idx < -1:
            LOGGER.warning(f"A joint with the name '{joint_name}' was not found")
        else:
            joint_index_dict[joint_name] = joint_idx

    return joint_index_dict
