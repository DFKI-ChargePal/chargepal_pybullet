import logging
import pybullet as p

# mypy
from typing import Tuple, List


LOGGER = logging.getLogger(__name__)


def draw_sphere_marker(
        position: Tuple[float, ...],
        radius: float, color: Tuple[float, ...],
        physics_client_id: int = 0
) -> int:

    """ Function to draw a sphere marker """
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=physics_client_id)
    marker_id: int = p.createMultiBody(
        basePosition=position,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vs_id,
        physicsClientId=physics_client_id
    )
    return marker_id


def extract_name(byte_name: bytes) -> str:
    return list(filter(None, str(byte_name).split("'")))[-1]


def link_name2link_idx(body_id: int, links: List[str], client_id: int) -> List[int]:
    index_list = []
    for link in links:
        found = False
        for idx in range(p.getNumJoints(bodyUniqueId=body_id, physicsClientId=client_id)):
            joint_info = p.getJointInfo(bodyUniqueId=body_id, jointIndex=idx, physicsClientId=client_id)
            link_name = extract_name(joint_info[12])
            if link_name == link:
                index_list.append(idx)
                found = True

        if not found:
            LOGGER.warning(f"A link with the name '{link}' was not found!")
    return index_list
