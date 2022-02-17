import pybullet as p

# mypy
from typing import Tuple


def draw_sphere_marker(position: Tuple[float, ...], radius: float, color: Tuple[float, ...], physics_client_id: int = 0) -> int:
    """ Function to draw a sphere marker """
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=physics_client_id)
    marker_id: int = p.createMultiBody(
        basePosition=position,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vs_id,
        physicsClientId=physics_client_id
    )
    return marker_id
