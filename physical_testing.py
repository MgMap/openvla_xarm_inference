"""
physical_testing.py

Enhanced client script for communicating with OpenVLA server deployment.
Includes proper preprocessing, normalization, and robot execution capabilities.

Dependencies:
    pip install requests json-numpy opencv-python numpy pillow pyserial

Usage:
    python physical_testing.py --server_url http://localhost:8000 --instruction "grasp the red object"
"""

import argparse
import json
import time
from typing import Optional, Dict, Tuple
import cv2
import json_numpy
import numpy as np
import requests
from PIL import Image
from dataclasses import dataclass

# Patch json to handle numpy arrays
json_numpy.patch()


@dataclass
class RobotState:
    """Current robot state including pose and gripper state."""
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [rx, ry, rz] in axis-angle or euler
    gripper: float        # gripper state (-1 to 1 or 0 to 1)


class XArm7Controller:
    """xArm7 robot controller interface."""
    
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
            'x': [150.0, 700.0],   # Approximate xArm7 reach
            'y': [-500.0, 500.0],
            'z': [10.0, 600.0]
        }
        
        # Safety constraints
        self.max_speed = 30.0  # mm/s
        self.max_acceleration = 100.0  # mm/s^2
        
    def connect(self):
        """Connect to xArm7 robot."""
        try:
            from xarm.wrapper import XArmAPI
            self.xarm = XArmAPI(self.robot_ip, is_radian=False)
            self.xarm.motion_enable(enable=True)
            self.xarm.set_mode(7)
            self.xarm.set_state(state=0)
            ret = self.xarm.get_position()
            if ret[0] == 0:
                current_pos = ret[1]
                self.current_state.position = np.array(current_pos[:3])
                self.current_state.rotation = np.array(current_pos[3:6])
            print(f"Connected to xArm7 at {self.robot_ip}")
            return True
            
        except ImportError:
            print("xArm SDK not installed. Install with: pip install xarm-python-sdk")
            print("Running in simulation mode...")
            return False
        except Exception as e:
            print(f"Failed to connect to xArm7: {e}")
            print("Running in simulation mode...")
            return False

    def disconnect(self):
        """Disconnect from xArm7."""
        if self.xarm is not None:
            self.xarm.disconnect()
            print("Disconnected from xArm7")
    
    def execute_action(self, action: np.ndarray) -> bool:
        """
        Execute absolute position action on xArm7.
        
        Args:
            action: 7-DoF action [x, y, z, rx, ry, rz, gripper] - ABSOLUTE positions
                   - Position in mm
                   - Rotation in degrees
                   - Gripper: 0-1 range (will be converted to 0-850 for xArm gripper)
        
        Returns:
            True if action was executed successfully
        """
        try:
            # Parse action - these are ABSOLUTE positions, not deltas
            target_pos = action[:3].copy()  # [x, y, z] in mm
            target_rot = action[3:6].copy()  # [rx, ry, rz] in degrees
            gripper_cmd = action[6]  # 0-1 range
            
            # Apply workspace limits for safety
            target_pos[0] = np.clip(target_pos[0], *self.workspace_limits['x'])
            target_pos[1] = np.clip(target_pos[1], *self.workspace_limits['y'])
            target_pos[2] = np.clip(target_pos[2], *self.workspace_limits['z'])
            
            # Limit rotation angles (typical xArm limits)
            target_rot = np.clip(target_rot, -180.0, 180.0)
            
            # Convert gripper command (0-1) to xArm gripper range (0-850)
            gripper_pos = np.clip(gripper_cmd * 850.0, 0.0, 850.0)
            
            # Execute movement
            if self.xarm is not None:
                ret = self.xarm.set_position(
                    x=target_pos[0], y=target_pos[1], z=target_pos[2],
                    roll=target_rot[0], pitch=target_rot[1], yaw=target_rot[2],
                    speed=self.max_speed,
                    wait=True,  # Wait for completion
                    # relative=False  # Absolute positioning
                )
                
                # Check if ret is a tuple/list or just an integer
                if isinstance(ret, (list, tuple)):
                    ret_code = ret[0]
                else:
                    ret_code = ret
                
                if ret_code != 0:  # Error occurred
                    print(f"xArm movement failed with code: {ret_code}")
                    return False
                
                # Control gripper if available
                try:
                    gripper_ret = self.xarm.set_gripper_position(gripper_pos, wait=True)
                    if isinstance(gripper_ret, (list, tuple)):
                        gripper_code = gripper_ret[0]
                    else:
                        gripper_code = gripper_ret
                    
                    if gripper_code != 0:
                        print(f"Gripper movement failed with code: {gripper_code}")
                except Exception as e:
                    print(f"Gripper control not available or failed: {e}")
                
                # Update current state
                self.current_state.position = target_pos
                self.current_state.rotation = target_rot
                self.current_state.gripper = gripper_cmd
                
                print(f"✓ Moved to: Pos=[{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}], "
                      f"Rot=[{target_rot[0]:.1f}, {target_rot[1]:.1f}, {target_rot[2]:.1f}], "
                      f"Gripper={gripper_cmd:.2f}")
                
            else:
                # Simulation mode
                self.current_state.position = target_pos
                self.current_state.rotation = target_rot
                self.current_state.gripper = gripper_cmd
                
                print(f"SIM: Pos=[{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}], "
                      f"Rot=[{target_rot[0]:.1f}, {target_rot[1]:.1f}, {target_rot[2]:.1f}], "
                      f"Gripper={gripper_cmd:.2f}")
                time.sleep(0.1)  # Simulate execution time
            
            return True
            
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
    
    def get_state(self) -> RobotState:
        """Get current robot state."""
        if self.xarm is not None:
            try:
                # Get current position from robot
                ret = self.xarm.get_position()
                if isinstance(ret, (list, tuple)) and ret[0] == 0:  # Success
                    current_pos = ret[1]
                    self.current_state.position = np.array(current_pos[:3])
                    self.current_state.rotation = np.array(current_pos[3:6])
                
                # Get gripper position
                try:
                    ret_gripper = self.xarm.get_gripper_position()
                    if isinstance(ret_gripper, (list, tuple)) and ret_gripper[0] == 0:
                        gripper_pos = ret_gripper[1]
                        self.current_state.gripper = gripper_pos / 850.0  # Convert to 0-1 range
                except Exception as e:
                    print(f"Could not get gripper state: {e}")
                    
            except Exception as e:
                print(f"Error getting robot state: {e}")
        
        return self.current_state
    
    def home(self):
        """Move robot to home position."""
        home_position = np.array([250.0, 0.0, 100.0, 180.0, 0.0, 0.0, 0.5])  # Safe home pose
        return self.execute_action(home_position)

class ActionPostprocessor:
    """Handle action postprocessing and normalization."""
    
    def __init__(self, action_scale: float = 1.0, gripper_threshold: float = 0.0, use_absolute: bool = False):
        """
        Initialize action postprocessor.
        
        Args:
            action_scale: Scale factor for position/rotation actions
            gripper_threshold: Threshold for binarizing gripper actions
            use_absolute: If True, treat actions as absolute positions; if False, treat as deltas
        """
        self.action_scale = action_scale
        self.gripper_threshold = gripper_threshold
        self.use_absolute = use_absolute
    
    def postprocess_action(self, raw_action: np.ndarray, current_state: RobotState,
                          unnorm_key: Optional[str] = None) -> np.ndarray:
        """
        Postprocess action from model output.
        
        Args:
            raw_action: Raw action from model (already denormalized by server)
            current_state: Current robot state
            unnorm_key: Dataset key used for denormalization
            
        Returns:
            Absolute action ready for robot execution [x, y, z, rx, ry, rz, gripper]
        """
        action = raw_action.copy()
        
        if self.use_absolute:
            # Treat actions as ABSOLUTE positions
            print("Using ABSOLUTE positioning mode")
            absolute_position = action[:3]   # Apply scaling to absolute positions
            absolute_rotation = action[3:6]  # Apply scaling to absolute rotations
        else:
            # Treat actions as DELTAS/RELATIVE movements (OpenVLA standard)
            print("Using DELTA/RELATIVE positioning mode")
            
            # Scale deltas
            position_deltas = action[:3] * self.action_scale  # mm
            rotation_deltas = action[3:6] * self.action_scale  # degrees
            
            # Convert to absolute coordinates by adding to current position
            absolute_position = current_state.position + position_deltas
            absolute_rotation = current_state.rotation + rotation_deltas
        gripper_cmd = np.clip(action[6], 0.0, 1.0)
        if abs(gripper_cmd - 0.5) > self.gripper_threshold:
            gripper_cmd = 1.0 if gripper_cmd > 0.5 else 0.0
        
        # Combine into absolute action
        absolute_action = np.concatenate([absolute_position, absolute_rotation, [gripper_cmd]])
        
        return absolute_action


class SafetyMonitor:
    def __init__(self, position_limits: Dict[str, Tuple[float, float]], max_velocity=100.0, max_delta=100.0):
        self.position_limits = position_limits
        self.max_velocity = max_velocity
        self.max_delta = max_delta  # Maximum allowed delta per step (mm)
        self.last_position = None
        self.last_time = None
        self.emergency_stop = False
    
    def check_safety(self, current_state: RobotState, 
                    proposed_action: np.ndarray, raw_deltas: Optional[np.ndarray] = None) -> bool:
        """
        Check if proposed action is safe to execute.
        
        Args:
            current_state: Current robot state
            proposed_action: Proposed absolute action [x, y, z, rx, ry, rz, gripper]
            raw_deltas: Optional raw deltas for additional safety checks
        
        Returns:
            True if action is safe, False otherwise
        """
        
        # Check workspace limits for absolute positions
        target_pos = proposed_action[:3]
        for i, axis in enumerate(['x', 'y', 'z']):
            if axis in self.position_limits:
                min_val, max_val = self.position_limits[axis]
                if not (min_val <= target_pos[i] <= max_val):
                    print(f"Safety violation: {axis} position {target_pos[i]:.1f} outside limits [{min_val}, {max_val}]")
                    return False
        
        # Check delta magnitude (how much we're moving in one step)
        # Only check delta magnitude if raw_deltas is not None
        if raw_deltas is not None:
            position_delta_magnitude = np.linalg.norm(raw_deltas[:3])
            if position_delta_magnitude > self.max_delta:
                print(f"Safety violation: position delta magnitude {position_delta_magnitude:.1f} mm exceeds limit {self.max_delta} mm")
                return False
        
        # Check velocity limits (distance between current and target position)
        distance = np.linalg.norm(target_pos - current_state.position)
        current_time = time.time()
        
        if self.last_position is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                velocity = distance / dt
                if velocity > self.max_velocity:
                    print(f"Safety violation: velocity {velocity:.1f} mm/s exceeds limit {self.max_velocity} mm/s")
                    return False
        
        self.last_position = current_state.position.copy()
        self.last_time = current_time
        return True

class ImagePreprocessor:
    """Handle image preprocessing to match training data distribution."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), center_crop: bool = True):
        self.target_size = target_size
        self.center_crop = center_crop
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to match OpenVLA training distribution.
        
        Args:
            image: Raw image from camera (H, W, 3) in RGB format
            
        Returns:
            Preprocessed image ready for model inference
        """
        # Convert to PIL Image for processing
        pil_image = Image.fromarray(image)
        
        if self.center_crop:
            # Apply center crop (90% area as used in training with image augmentations)
            width, height = pil_image.size
            crop_scale = 0.9
            crop_size = int(min(width, height) * crop_scale)
            
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            
            pil_image = pil_image.crop((left, top, right, bottom))
        
        # Resize to target size
        resized_image = pil_image.resize(self.target_size, Image.LANCZOS)
        
        # Convert back to numpy array
        processed_image = np.array(resized_image)
        processed_image = processed_image.astype(np.uint8)
        
        return processed_image


class OpenVLAClient:
    def __init__(self, server_url: str = "http://10.136.109.82:8000", 
                 camera_index: int = 4, robot_ip: str = "192.168.1.239", 
                 use_absolute: bool = False):
        """
        Initialize the enhanced OpenVLA client for xArm7.
        
        Args:
            server_url: URL of the OpenVLA server
            camera_index: Camera index to capture images from
            robot_ip: IP address of xArm7 robot
            use_absolute: If True, treat actions as absolute positions; if False, treat as deltas
        """
        self.server_url = server_url.rstrip('/')
        self.camera_index = camera_index
        self.cap = None
        self.use_absolute = use_absolute
        
        # Initialize components for xArm7
        # Use center_crop=True if model was trained with image augmentations
        self.preprocessor = ImagePreprocessor(center_crop=True)  
        
        # Adjust action scale based on positioning mode
        if use_absolute:
            # For absolute positioning, might need different scaling
            action_scale = 1.0  # Start with no scaling for absolute mode
            print("Initialized in ABSOLUTE positioning mode")
        else:
            # For delta positioning, use conservative scaling
            action_scale = 0.05  # Small scale for relative movements
            print("Initialized in DELTA/RELATIVE positioning mode")
        
        self.postprocessor = ActionPostprocessor(
            action_scale=action_scale, 
            use_absolute=use_absolute
        )
        self.robot = XArm7Controller(robot_ip)
        self.safety = SafetyMonitor({'x': [150.0, 650.0], 'y': [-500.0, 500.0], 'z': [10.0, 400.0]})
        self._setup_camera()

    def _setup_camera(self):
        """Initialize the camera."""
        print(f"Setting up camera with index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera setup successful!")
    
    def connect_robot(self) -> bool:
        """Connect to xArm7 robot."""
        return self.robot.connect()
    
    def capture_and_preprocess_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture and preprocess image from camera.
        
        Returns:
            Tuple of (raw_image, preprocessed_image)
        """
        if self.cap is None:
            raise RuntimeError("Camera not initialized")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
        
        # Convert BGR to RGB
        raw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        preprocessed_image = self.preprocessor.preprocess_image(raw_image)
        
        return raw_image, preprocessed_image
    
    def predict_action(self, image: np.ndarray, instruction: str, 
                      unnorm_key: Optional[str] = None) -> np.ndarray:
        """
        Send image and instruction to OpenVLA server and get predicted action.
        """
        payload = {
            "image": image,
            "instruction": instruction
        }
        
        if unnorm_key is not None:
            payload["unnorm_key"] = unnorm_key
        
        try:
            response = requests.post(
                f"{self.server_url}/act",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            action = response.json()
            
            if isinstance(action, str) and action == "error":
                raise RuntimeError("Server returned an error")
            
            return np.array(action)
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with server: {e}")
    
    def execute_task(self, instruction: str, max_steps: int = 100, 
                    control_frequency: float = 5.0, unnorm_key: Optional[str] = None,
                    display_images: bool = True, save_data: bool = False):
        """
        Execute a task with full robot control pipeline.
        
        Args:
            instruction: Task instruction
            max_steps: Maximum number of control steps
            control_frequency: Control frequency in Hz
            unnorm_key: Dataset key for action denormalization
            display_images: Whether to display camera feed
            save_data: Whether to save execution data
        """
        print(f"Starting task execution: '{instruction}'")
        print(f"Max steps: {max_steps}, Control frequency: {control_frequency} Hz")
        
        if self.use_absolute:
            print("Using ABSOLUTE positioning mode")
        else:
            print("Using DELTA/RELATIVE positioning mode (OpenVLA standard)")
        
        step_duration = 1.0 / control_frequency
        execution_data = []
        
        # Connect to robot
        robot_connected = self.connect_robot()
        if not robot_connected:
            print("WARNING: Robot not connected, running in simulation mode")
        
        # Move to home position
        print("Moving to home position...")
        self.robot.home()
        time.sleep(1.0)
        
        successful_steps = 0
        
        for step in range(max_steps):
            step_start_time = time.time()
            
            try:
                # Check for emergency stop
               
                
                # Capture and preprocess image
                raw_image, processed_image = self.capture_and_preprocess_image()
                
                # Get current robot state
                robot_state = self.robot.get_state()
                
                # Display image if requested
                if display_images:
                    display_img = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                    cv2.putText(display_img, f"Step: {step+1}/{max_steps}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_img, f"Task: {instruction}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_img, f"Pos: [{robot_state.position[0]:.1f}, "
                              f"{robot_state.position[1]:.1f}, {robot_state.position[2]:.1f}]", 
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Show positioning mode
                    mode_text = "ABSOLUTE" if self.use_absolute else "DELTA"
                    cv2.putText(display_img, f"Mode: {mode_text}", (10, 150), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                    cv2.imshow("OpenVLA xArm7 Control", display_img)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Stopping due to user input...")
                        break
                    elif key == ord(' '):  # Space bar for emergency stop
                        self.safety.trigger_emergency_stop()
                        self.robot.emergency_stop()
                        continue
                
                # Get action prediction from server
                raw_action = self.predict_action(processed_image, instruction, unnorm_key)
                print(f"DEBUG - Raw action from server: {raw_action}")
                # Postprocess action based on positioning mode
                absolute_action = self.postprocessor.postprocess_action(
                    raw_action, robot_state, unnorm_key
                )
                
                # Calculate deltas for safety and debugging
                actual_deltas = absolute_action[:3] - robot_state.position

                # Safety check on absolute target position and deltas
                # Only check max_delta in DELTA mode
                if self.use_absolute:
                    safe = self.safety.check_safety(robot_state, absolute_action, raw_deltas=None)
                else:
                    safe = self.safety.check_safety(robot_state, absolute_action, actual_deltas)

                if not safe:
                    print(f"Step {step+1}: Safety check failed, skipping action")
                    continue
                
                # Execute absolute action on robot
                execution_success = self.robot.execute_action(absolute_action)
                
                if execution_success:
                    successful_steps += 1
                    # Show debugging info based on mode
                    if self.use_absolute:
                        print(f"Step {step+1:6d}: ✓ Raw Action = {raw_action[:6]}")
                        print(f"             Absolute Target = {absolute_action[:6]}")
                        print(f"             Movement Delta = {actual_deltas}")
                    else:
                        print(f"Step {step+1:6d}: ✓ Raw Delta = {raw_action[:6]}")
                        print(f"             Scaled Delta = {actual_deltas}")
                        print(f"             Target = {absolute_action[:6]}")
                else:
                    print(f"Step {step+1:6d}: ✗ Failed to execute action")
                
                # Save execution data if requested
                if save_data:
                    execution_data.append({
                        'step': step,
                        'timestamp': time.time(),
                        'positioning_mode': 'absolute' if self.use_absolute else 'delta',
                        'raw_image': raw_image,
                        'processed_image': processed_image,
                        'robot_state': {
                            'position': robot_state.position.tolist(),
                            'rotation': robot_state.rotation.tolist(),
                            'gripper': robot_state.gripper
                        },
                        'raw_action': raw_action.tolist(),
                        'actual_deltas': actual_deltas.tolist(),
                        'absolute_action': absolute_action.tolist(),
                        'execution_success': execution_success
                    })
                
                # Maintain control frequency
                elapsed_time = time.time() - step_start_time
                if elapsed_time < step_duration:
                    time.sleep(step_duration - elapsed_time)
                
            except KeyboardInterrupt:
                print("\nStopping due to keyboard interrupt...")
                break
            except Exception as e:
                print(f"Error at step {step+1}: {e}")
                break
        
        # Save execution data
        if save_data and execution_data:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            mode_suffix = "absolute" if self.use_absolute else "delta"
            data_file = f"xarm7_execution_data_{mode_suffix}_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_data = []
            for item in execution_data:
                json_item = item.copy()
                json_item['raw_image'] = item['raw_image'].tolist()
                json_item['processed_image'] = item['processed_image'].tolist()
                json_data.append(json_item)
            
            with open(data_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved execution data to: {data_file}")
        
        print(f"Task execution completed!")
        print(f"Successful steps: {successful_steps}/{step+1}")
        print(f"Positioning mode used: {'ABSOLUTE' if self.use_absolute else 'DELTA'}")
        
        # Disconnect robot
        self.robot.disconnect()
    
    def test_single_prediction(self, instruction: str, unnorm_key: Optional[str] = None,
                             save_image: bool = False):
        """Test a single prediction without robot execution."""
        print(f"Testing single prediction for: '{instruction}'")
        print(f"Positioning mode: {'ABSOLUTE' if self.use_absolute else 'DELTA'}")
        
        # Capture and preprocess image
        raw_image, processed_image = self.capture_and_preprocess_image()
        
        # Save images if requested
        if save_image:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            mode_suffix = "absolute" if self.use_absolute else "delta"
            Image.fromarray(raw_image).save(f"raw_image_{mode_suffix}_{timestamp}.jpg")
            Image.fromarray(processed_image).save(f"processed_image_{mode_suffix}_{timestamp}.jpg")
            print(f"Saved images with timestamp {timestamp}")
        
        # Get prediction
        try:
            # Get current robot state
            robot_state = self.robot.get_state()
            
            raw_action = self.predict_action(processed_image, instruction, unnorm_key)
            absolute_action = self.postprocessor.postprocess_action(raw_action, robot_state, unnorm_key)
            
            # Calculate movement deltas
            actual_deltas = absolute_action[:3] - robot_state.position
            
            # print(f"DEBUG - Raw action from server:")
            # print(f"  Shape: {raw_action.shape}")
            # print(f"  Values: {raw_action}")
            # print(f"  Min/Max: [{raw_action.min():.3f}, {raw_action.max():.3f}]")
            # print(f"")
            # print(f"Positioning Mode: {'ABSOLUTE' if self.use_absolute else 'DELTA'}")
            # print(f"Raw action: {raw_action}")
            # print(f"Processed action: {absolute_action}")
            # print(f"Action breakdown:")
            
            # if self.use_absolute:
            #     print(f"  Raw Position (absolute): [{raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}]")
            #     print(f"  Scaled Position (absolute): [{absolute_action[0]:.3f}, {absolute_action[1]:.3f}, {absolute_action[2]:.3f}] mm")
            #     print(f"  Movement Delta: [{actual_deltas[0]:.3f}, {actual_deltas[1]:.3f}, {actual_deltas[2]:.3f}] mm")
            # else:
            #     print(f"  Position Delta (raw): [{raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}]")
            #     print(f"  Position Delta (scaled): [{actual_deltas[0]:.3f}, {actual_deltas[1]:.3f}, {actual_deltas[2]:.3f}] mm")
            
            # print(f"  Rotation: [{raw_action[3]:.3f}, {raw_action[4]:.3f}, {raw_action[5]:.3f}]")
            # print(f"  Current Position: [{robot_state.position[0]:.1f}, {robot_state.position[1]:.1f}, {robot_state.position[2]:.1f}] mm")
            # print(f"  Target Position: [{absolute_action[0]:.1f}, {absolute_action[1]:.1f}, {absolute_action[2]:.1f}] mm")
            # print(f"  Target Rotation: [{absolute_action[3]:.1f}, {absolute_action[4]:.1f}, {absolute_action[5]:.1f}] deg")
            # print(f"  Gripper: {absolute_action[6]:.3f}")
            
            # # Display image with action overlay
            # display_img = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            # cv2.putText(display_img, f"Instruction: {instruction}", (10, 30), 
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # mode_text = "ABSOLUTE" if self.use_absolute else "DELTA"
            # cv2.putText(display_img, f"Mode: {mode_text}", (10, 70), 
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # if self.use_absolute:
            #     cv2.putText(display_img, f"Target: {absolute_action[:3]}", (10, 100), 
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            #     cv2.putText(display_img, f"Delta: {actual_deltas}", (10, 120), 
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            # else:
            #     cv2.putText(display_img, f"Delta: {actual_deltas}", (10, 100), 
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            #     cv2.putText(display_img, f"Target: {absolute_action[:3]}", (10, 120), 
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # cv2.imshow("Single Prediction Test - xArm7", display_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during prediction: {e}")

    def cleanup(self):
        """Release all resources."""
        if self.cap is not None:
            self.cap.release()
        self.robot.disconnect()
        cv2.destroyAllWindows()
        print("All resources released.")


def main():
    parser = argparse.ArgumentParser(description="OpenVLA xArm7 Robot Control Client")
    parser.add_argument("--server_url", type=str, default="http://10.136.109.82:8000",
                       help="URL of the OpenVLA server")
    parser.add_argument("--camera_index", type=int, default=4,
                       help="Camera index to use for image capture")
    parser.add_argument("--robot_ip", type=str, default="192.168.1.239",
                       help="IP address of xArm7 robot")
    parser.add_argument("--instruction", type=str, required=True,
                       help="Task instruction for the robot")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig",
                       help="Dataset key for action denormalization")
    parser.add_argument("--mode", type=str, choices=["single", "execute"], default="single",
                       help="Mode: 'single' for prediction test, 'execute' for full task execution")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Maximum number of steps for task execution")
    parser.add_argument("--control_frequency", type=float, default=5.0,
                       help="Control frequency in Hz")
    parser.add_argument("--no_display", action="store_true",
                       help="Disable image display")
    parser.add_argument("--save_image", action="store_true",
                       help="Save captured images")
    parser.add_argument("--save_data", action="store_true",
                       help="Save execution data")
    
    # New positioning mode arguments
    positioning_group = parser.add_mutually_exclusive_group()
    positioning_group.add_argument("--absolute", action="store_true",
                                 help="Treat model outputs as absolute positions (direct coordinates)")
    positioning_group.add_argument("--delta", action="store_true",
                                 help="Treat model outputs as delta/relative movements (OpenVLA standard)")
    
    args = parser.parse_args()
    
    # Determine positioning mode
    if args.absolute:
        use_absolute = True
        print("Using ABSOLUTE positioning mode")
    elif args.delta:
        use_absolute = False
        print("Using DELTA positioning mode")
    else:
        # Default to delta mode (OpenVLA standard)
        use_absolute = False
        print("Using DELTA positioning mode (default)")
    
    # Initialize client
    client = OpenVLAClient(
        server_url=args.server_url, 
        camera_index=args.camera_index,
        robot_ip=args.robot_ip,
        use_absolute=use_absolute
    )
    
    try:
        if args.mode == "single":
            # Single prediction test
            client.test_single_prediction(
                instruction=args.instruction,
                unnorm_key=args.unnorm_key,
                save_image=args.save_image
            )
        else:
            # Full task execution
            client.execute_task(
                instruction=args.instruction,
                max_steps=args.max_steps,
                control_frequency=args.control_frequency,
                unnorm_key=args.unnorm_key,
                display_images=not args.no_display,
                save_data=args.save_data
            )
    
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()