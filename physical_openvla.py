import argparse
import json
import os
from typing import Optional, Dict, _testimportmultiple

import cv2
import json_numpy
import numpy as np
import requests
from PIL import Image
from dataclasses import dataclass
from xarm.wrapper import XArmAPI

json_numpy.patch()

class RobotState:
    """current state of the robot including position, rotation, and gripper status."""
    position: np.ndarray # [x,y,z]
    rotation: np.ndarray # [roll, pitch, yaw]
    gripper_status: float # [0, 1] (0: open, 1: closed)

class XArm7Controller:
    """Controller for the XArm7 robot."""
    
    def __init__(self, robot_ip: str = "192.168.1.239", port: int = 8000):
        """Initialize xArm7 controller."""
        self.robot_ip = robot_ip
        self.port = port
        self.xarm = None
        self.current_state = RobotState(
            position=np.array([250.0, 0.0, 100.0]),  # Default home position in mm
            rotation=np.array([180.0, 0.0, 0.0]),   # Default orientation in degrees
            gripper=800.0  # Default gripper position (0-850 for xArm gripper)
        )
        
        # xArm7 workspace limits (in mm)
        self.workspace_limits = {
            'x': [150.0, 800.0],   # Approximate xArm7 reach
            'y': [-500.0, 500.0],
            'z': [10.0, 400.0]
        }
        
        # Safety constraints
        self.max_speed = 100.0  # mm/s
        self.max_acceleration = 1000.0  # mm/s^2

    def connect(self):
        """connect to the xArm7 robot."""

        try:
            from xarm.wrapper import XArmAPI
            self.xarm = ZArmAPI(self.robot_ip, is_radian=False)
            self.xarm.motion_enable(enable=True)
            self.xarm.set_mode(0)  # Set to joint mode
            self.xarm.set_state(state=0)  # Set to idle state

            ret = self.xarm.get_position()
            if ret[0] == [0]:
                current_pos = ret[1]
                self.current_state.position = np.array(current_pos[:3])
                self.current_state.rotation = np.array(current_pos[3:6])

            print(f"Connected to xArm7 at {self.robot_ip}")

            return True
        except ImportError:
            print("xArm SDK is not installed")
            return False
        except Exception as e:
            print(f"Failed to connect to xArm7: {e}")
            return False
        
    def disconnect(self):
        """Disconnect from the xArm7 robot."""
        if self.xarm:
            self.xarm.disconnect()
            print("Disconnected from xArm7")
        else:
            print("No active connection to xArm7")

    def execute_action(self, action: Dict[str, float]) -> bool:
        """Execute an action on the xArm7 robot."""
        if not self.xarm:
            print("Not connected to xArm7")
            return False
        
        # Validate action
        if 'position' in action:
            position = np.array(action['position'])
            if not self.is_within_workspace(position):
                print("Position out of workspace limits")
                return False
            
            self.xarm.set_position(*position, speed=self.max_speed, wait=True)
            self.current_state.position = position
        
        if 'rotation' in action:
            rotation = np.array(action['rotation'])
            self.xarm.set_position(*self.current_state.position, roll=rotation[0], pitch=rotation[1], yaw=rotation[2], speed=self.max_speed, wait=True)
            self.current_state.rotation = rotation
        
        if 'gripper' in action:
            gripper_pos = action['gripper']
            if 0 <= gripper_pos <= 850:
                self.xarm.set_gripper_position(gripper_pos, speed=self.max_speed, wait=True)
                self.current_state.gripper = gripper_pos
            else:
                print("Gripper position out of range (0-850)")
                return False
        
        return True