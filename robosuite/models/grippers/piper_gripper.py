"""
Gripper model for AgileX Piper with the original Piper finger meshes.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class PiperGripperBase(GripperModel):
    """
    Piper two-finger gripper using link7/link8 meshes for visualization.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/piper_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        opening = 0.035 * 59.4 / 70.0
        return np.array([opening, -opening])

    @property
    def init_action(self):
        """Normalized two-actuator target matching the physical initial qpos."""
        opening = self.init_qpos[0]
        half_range = 0.035 / 2.0
        return np.array([opening / half_range - 1.0, -opening / half_range + 1.0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_finger_mesh_collision", "left_finger_collision", "left_fingerpad_collision"],
            "right_finger": ["right_finger_mesh_collision", "right_finger_collision", "right_fingerpad_collision"],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }


class PiperGripper(PiperGripperBase):
    """
    1-DoF open / close wrapper for Piper's symmetric slide fingers.
    """

    def format_action(self, action):
        """
        Maps continuous action into symmetric finger targets.
        -1 opens the gripper, +1 closes it.
        """
        assert len(action) == self.dof
        if np.asarray(self.current_action).size != 2:
            self.current_action = self.init_action.copy()
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.35

    @property
    def dof(self):
        return 1
