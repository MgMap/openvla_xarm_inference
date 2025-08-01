"""
Simple OpenVLA inference with xArm7 robot.
Usage: python simple_xarm_inference.py --instruction "pick up the red block"
"""

import argparse
import cv2
import numpy as np
import requests
import time
from PIL import Image
from typing import Optional, Dict, Tuple
import requests
from dataclasses import dataclass
import json  # Custom patch to handle numpy arrays in JSON
import json_numpy
json_numpy.patch()

class SimpleXArmOpenVLA:
    def __init__(self, server_url="http://localhost:8000", robot_ip="192.168.1.239", camera_index=12):
        self.server_url = server_url
        self.robot_ip = robot_ip
        self.camera_index = camera_index
        self.xarm = None
        self.cap = None
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize robot
        self.init_robot()
    
    def init_robot(self):
        """Initialize xArm7 robot connection."""
        try:
            from xarm.wrapper import XArmAPI
            self.xarm = XArmAPI(self.robot_ip)
            self.xarm.motion_enable(enable=True)
            self.xarm.set_mode(0)  # Servo mode
            self.xarm.set_state(state=0)
            self.xarm.set_gripper_enable(enable=True)  # Enable gripper
            self.xarm.set_gripper_position(0)  # Open gripper
            self.xarm.set_position(x=250, y=0, z=100, roll = 180, pitch = 0, yaw = 90, speed=30, mvacc=100)  # Move to home position
            print(f"✓ Connected to xArm7 at {self.robot_ip}")
        except Exception as e:
            print(f"⚠ Robot connection failed: {e}")
            print("Running in simulation mode")
    
    def capture_image(self):
        """Capture image from camera."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        
        # Convert BGR to RGB and resize
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (224, 224))
        
        return image

    def predict_action(self, image: np.ndarray, instruction: str, unnorm_key: Optional[str] = None):
        """Get action prediction from OpenVLA server."""
        payload = {
            "image": image,
            "instruction": instruction
        }

        if unnorm_key is not None:
            payload["unnorm_key"] = unnorm_key
        
        response = requests.post(f"{self.server_url}/act", json=payload, timeout=10)
        response.raise_for_status()
        
        action = np.array(response.json())
        return action
    
    def execute_action(self, action):
        """Execute action on xArm7."""
        # Extract position, rotation, and gripper from action
        action_scale = 50 # Scale factor for action values
        x, y, z = action[0], action[1], action[2]
        rx, ry, rz = action[3], action[4], action[5]
        gripper = action[6]
        
        # Scale action (adjust these values based on your setup)

        x = x * action_scale  # Scale and offset for xArm workspace
        y = y * action_scale
        z = z * action_scale
        
        # Apply safety limits
        

        code, current_position = self.xarm.get_position()
        target_x = current_position[0] + x
        target_y = current_position[1] + y
        target_z = current_position[2] + z
        target_rx = current_position[3] + rx
        target_ry = current_position[4] + ry
        target_rz = current_position[5] + rz

        target_x = np.clip(target_x, 100, 600)
        target_y = np.clip(target_y, -400, 400)
        target_z = np.clip(target_z, -1.4, 400)
        # Get current position for debugging
        print(f"Current position: x={current_position[0]:.1f}, y={current_position[1]:.1f}, z={current_position[2]:.1f}, rx = {current_position[3]:.1f}, ry = {current_position[4]:.1f}, rz = {current_position[5]:.1f}")
        print(f"Target position: x={target_x:.1f}, y={target_y:.1f}, z={target_z:.1f}, rx = {target_rx:.1f}, ry = {target_ry:.1f}, rz = {target_rz:.1f}")

        # print(f"Moving to: x={x:.1f}, y={y:.1f}, z={z:.1f}")
        
        if self.xarm:
            # Move robot
            ret = self.xarm.set_position(target_x, target_y, target_z, target_rx, target_ry, target_rz, speed=75, wait=True)

            # Control gripper
            if gripper > 0.5:
                self.xarm.set_gripper_position(800, wait=True)  # Close
            else:
                self.xarm.set_gripper_position(0, wait=True)    # Open
                
            return ret == 0
        else:
            # Simulation mode
            print(f"SIM: Position=({x:.1f}, {y:.1f}, {z:.1f}), Gripper={gripper:.2f}")
            time.sleep(0.1)
            return True
    
    def run_inference(self, instruction, max_steps=1000):
        """Run inference loop."""
        print(f"Starting task: {instruction}")
        
        for step in range(max_steps):
            try:
                # Capture image
                image = self.capture_image()
                image = np.array(image, dtype=np.uint8)
                # # Show image
                # display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.putText(display_img, f"Step {step+1}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.imshow("OpenVLA xArm7", display_img)
                
                # # Check for quit
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                # Get action from OpenVLA
                unnorm_key = "ucsd_pick_and_place_dataset_converted_externally_to_rlds"
                action = self.predict_action(image, instruction, unnorm_key=unnorm_key)
                print(f"Step {step+1}: Action = {action}")
                
                # Execute action
                success = self.execute_action(action)
                if not success:
                    print("Action execution failed!")
                    break
                
                time.sleep(0.2)  # Control frequency
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        print("Task completed!")
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.xarm:
            self.xarm.disconnect()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default="http://10.136.109.82:8000", help="OpenVLA server URL")
    parser.add_argument("--robot_ip", default="192.168.1.239", help="xArm7 IP address")
    parser.add_argument("--camera_index", type=int, default=12, help="Camera index")
    parser.add_argument("--instruction", required=True, help="Task instruction")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps")
    
    args = parser.parse_args()
    
    # Create and run inference
    robot = SimpleXArmOpenVLA(
        server_url=args.server_url,
        robot_ip=args.robot_ip,
        camera_index=args.camera_index
    )
    robot.init_robot()
    ret = robot.xarm.set_position(x=250, y=0, z=100, roll = 180, pitch = 0, yaw = 90, speed=30, mvacc=100, wait=True)  # Move to home position
    
    try:
        robot.run_inference(args.instruction, args.max_steps)
    finally:
        robot.cleanup()

if __name__ == "__main__":
    main()