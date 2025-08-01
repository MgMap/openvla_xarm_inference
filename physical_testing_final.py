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
import os  # Add this import
from typing import Optional, Dict, Tuple, List  # Add List to the import
import cv2
import json_numpy
import numpy as np
import requests
from PIL import Image
from dataclasses import dataclass
import threading
import queue

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
            rotation=np.array([180.0, 0.0, 90.0]),   # Default orientation in degrees
            gripper=800.0  # Default gripper position (0-850 for xArm gripper)
        )
        
        # xArm7 workspace limits (in mm)
        self.workspace_limits = {
            'x': [100.0, 700.0],   # Approximate xArm7 reach
            'y': [-500.0, 500.0],
            'z': [-11.2, 500.0]
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
            self.xarm.set_mode(0)
            self.xarm.set_state(state=0)
            self.xarm.set_gripper_enable(enable=True)
            self.xarm.set_gripper_position(850, wait=True)  # Set gripper to open position
            ret = self.xarm.get_position()
            # ret = [0]  # Simulated success
            if ret[0] == 0:
                current_pos = ret[1]
                current_pos = [250.0, 0.0, 100.0, 180.0, 0.0, 0.0]  # Simulated position
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
                
                print(f"Moved to: Pos=[{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}], "
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
        home_position = np.array([250.0, 0.0, 100.0, 180.0, 0.0, 90.0, 0.5])  # Safe home pose
        return self.execute_action(home_position)
@dataclass
class EpisodeData:
    """Data structure for storing episode information."""
    episode_id: str
    instruction: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[Dict] = None
    success: Optional[bool] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []

class EpisodeManager:
    """Manages episode data collection and saving."""
    
    def __init__(self, save_dir: str = "episode_data"):
        self.save_dir = save_dir
        self.current_episode = None
        self.episode_counter = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
    def start_episode(self, instruction: str) -> str:
        """Start a new episode."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.episode_counter += 1
        episode_id = f"episode_{self.episode_counter:03d}_{timestamp}"
        
        self.current_episode = EpisodeData(
            episode_id=episode_id,
            instruction=instruction,
            start_time=time.time()
        )
        
        # Create episode-specific image directory
        self.episode_image_dir = os.path.join(self.save_dir, episode_id, "images")
        os.makedirs(self.episode_image_dir, exist_ok=True)
        
        print(f"Started episode: {episode_id}")
        print(f"Instruction: {instruction}")
        return episode_id
    
    def add_step(self, step_data: Dict):
        """Add a step to the current episode with efficient image storage."""
        if self.current_episode is None:
            raise RuntimeError("No active episode. Call start_episode() first.")
        
        # Extract and save images separately
        step_data_copy = step_data.copy()
        
        # Save raw image
        # if 'raw_image' in step_data_copy:
        #     raw_image_filename = f"step_{step_data['step']:04d}_raw.jpg"
        #     raw_image_path = os.path.join(self.episode_image_dir, raw_image_filename)
        #     Image.fromarray(step_data_copy['raw_image']).save(raw_image_path, quality=85)
        #     step_data_copy['raw_image'] = raw_image_filename  # Store filename instead
        
        # Save processed image  
        # if 'processed_image' in step_data_copy:
        #     processed_image_filename = f"step_{step_data['step']:04d}_processed.jpg"
        #     processed_image_path = os.path.join(self.episode_image_dir, processed_image_filename)
        #     Image.fromarray(step_data_copy['processed_image']).save(processed_image_path, quality=85)
        #     step_data_copy['processed_image'] = processed_image_filename
            
        # Save VLA input image (if different from processed)
        if 'vla_input' in step_data_copy and 'image' in step_data_copy['vla_input']:
            vla_input_filename = f"step_{step_data['step']:04d}_vla_input.jpg"
            vla_input_path = os.path.join(self.episode_image_dir, vla_input_filename)
            Image.fromarray(step_data_copy['vla_input']['image']).save(vla_input_path, quality=85)
            step_data_copy['vla_input']['image'] = vla_input_filename
        
        # Convert remaining numpy arrays to lists for JSON serialization
        serializable_data = self._make_serializable(step_data_copy)
        self.current_episode.steps.append(serializable_data)
    
    def end_episode(self, manual_success_input: bool = True) -> bool:
        """End the current episode and optionally get manual success label."""
        if self.current_episode is None:
            raise RuntimeError("No active episode to end.")
        
        self.current_episode.end_time = time.time()
        duration = self.current_episode.end_time - self.current_episode.start_time
        
        print(f"\nEpisode {self.current_episode.episode_id} completed!")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Total steps: {len(self.current_episode.steps)}")
        print(f"Instruction: {self.current_episode.instruction}")
        
        # Get manual success label
        if manual_success_input:
            success = self._get_manual_success_label()
            self.current_episode.success = success
            
            # Optional notes
            notes = input("Add any notes about this episode (optional): ").strip()
            if notes:
                self.current_episode.notes = notes
        
        # Save episode data
        self._save_episode()
        
        # Reset current episode
        episode_success = self.current_episode.success
        self.current_episode = None
        
        return episode_success if episode_success is not None else False
    
    def _get_manual_success_label(self) -> bool:
        """Get manual success label from user input."""
        while True:
            try:
                response = input("\nDid the robot successfully complete the task? (y/n): ").strip().lower()
                if response in ['y', 'yes', '1', 'true']:
                    print("✓ Episode marked as SUCCESS")
                    return True
                elif response in ['n', 'no', '0', 'false']:
                    print("✗ Episode marked as FAILURE")
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except KeyboardInterrupt:
                print("\nDefaulting to failure due to interrupt.")
                return False
    
    def _make_serializable(self, data: Dict) -> Dict:
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        serializable = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable[key] = {
                    'type': 'numpy_array',
                    'data': value.tolist(),
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, dict):
                serializable[key] = self._make_serializable(value)
            elif isinstance(value, (list, tuple)):
                serializable[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item 
                    for item in value
                ]
            else:
                serializable[key] = value
        
        return serializable
    
    def _save_episode(self):
        """Save the current episode to a JSON file."""
        if self.current_episode is None:
            return
        
        filename = f"{self.current_episode.episode_id}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert episode to dictionary
        episode_dict = {
            'episode_id': self.current_episode.episode_id,
            'instruction': self.current_episode.instruction,
            'start_time': self.current_episode.start_time,
            'end_time': self.current_episode.end_time,
            'duration': self.current_episode.end_time - self.current_episode.start_time if self.current_episode.end_time else None,
            'success': self.current_episode.success,
            'notes': self.current_episode.notes,
            'total_steps': len(self.current_episode.steps),
            'image_directory': os.path.join(self.current_episode.episode_id, "images"),  # Relative path to images
            'steps': self.current_episode.steps
        }
        
        # Save to file with compression
        with open(filepath, 'w') as f:
            json.dump(episode_dict, f, indent=2)
        
        # Print file size info
        json_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        image_dir_size = self._get_directory_size(self.episode_image_dir) / (1024 * 1024)  # MB
        
        success_text = "SUCCESS" if self.current_episode.success else "FAILURE"
        print(f"Episode data saved to: {filepath}")
        print(f"JSON size: {json_size:.1f} MB, Images: {image_dir_size:.1f} MB")
        print(f"Result: {success_text}")
    
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size

    # Add method to load episode with images
    def load_episode(self, episode_file: str) -> Dict:
        """Load episode data and reconstruct images."""
        with open(episode_file, 'r') as f:
            episode_data = json.load(f)
        
        episode_dir = os.path.dirname(episode_file)
        image_dir = os.path.join(episode_dir, episode_data['image_directory'])
        
        # Reconstruct images for each step
        for step in episode_data['steps']:
            if 'raw_image' in step and isinstance(step['raw_image'], str):
                image_path = os.path.join(image_dir, step['raw_image'])
                if os.path.exists(image_path):
                    step['raw_image'] = np.array(Image.open(image_path))
            
            if 'processed_image' in step and isinstance(step['processed_image'], str):
                image_path = os.path.join(image_dir, step['processed_image'])
                if os.path.exists(image_path):
                    step['processed_image'] = np.array(Image.open(image_path))
                    
            if 'vla_input' in step and 'image' in step['vla_input'] and isinstance(step['vla_input']['image'], str):
                image_path = os.path.join(image_dir, step['vla_input']['image'])
                if os.path.exists(image_path):
                    step['vla_input']['image'] = np.array(Image.open(image_path))
        
        return episode_data
@dataclass
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


class ThreadedCamera:
    """Threaded camera capture to reduce latency."""
    
    def __init__(self, camera_index: int = 6):
        self.camera_index = camera_index
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue
        self.capture_thread = None
        self.running = False
        
    def start(self):
        """Start the camera capture thread."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
        
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print("Threaded camera started!")
    
    def _capture_frames(self):
        """Continuously capture frames in background thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Clear old frames and add new one
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                time.sleep(0.01)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the camera capture thread."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.cap:
            self.cap.release()

# Update OpenVLAClient to use threaded camera
class OpenVLAClient:
    def __init__(self, server_url: str = "http://10.136.109.82:8000", 
                 camera_index: int = 4, robot_ip: str = "192.168.1.239", 
                 use_absolute: bool = False, use_threaded_camera: bool = True, episode_save_dir: str = "episode_data"):
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
        self.use_threaded_camera = use_threaded_camera
        
        # Initialize components for xArm7
        # Initialize episode manager
        self.episode_manager = EpisodeManager(episode_save_dir)
        # Use center_crop=True if model was trained with image augmentations
        self.preprocessor = ImagePreprocessor(center_crop=True)  
        
        # Adjust action scale based on positioning mode
        if use_absolute:
            # For absolute positioning, might need different scaling
            action_scale = 1.0  # Start with no scaling for absolute mode
            print("Initialized in ABSOLUTE positioning mode")
        else:
            # For delta positioning, use conservative scaling
            action_scale = 20.0 # Small scale for relative movements
            print("Initialized in DELTA/RELATIVE positioning mode")
        
        self.postprocessor = ActionPostprocessor(
            action_scale=action_scale, 
            use_absolute=use_absolute
        )
        self.robot = XArm7Controller(robot_ip)
        self.safety = SafetyMonitor({'x': [100.0, 650.0], 'y': [-500.0, 500.0], 'z': [-11.2, 400.0]})

        if use_threaded_camera:
            self.threaded_camera = ThreadedCamera(camera_index)
            self.cap = None  # Don't use regular cap
        else:
            self.threaded_camera = None
            self._setup_camera()

    def _setup_camera(self):
        """Initialize the camera with optimized settings for low latency."""
        print(f"Setting up camera with index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
        
        # Optimize camera settings for low latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Critical latency reduction settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimum
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG
        
        # Additional optimization settings (may not work on all cameras)
        # try:
        #     self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        #     self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Fast exposure
        #     self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        #     self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
        # except:
        #     print("Some camera optimization settings not supported")
        
        # Flush initial frames to ensure fresh capture
        for _ in range(5):
            self.cap.read()
        
        print("Camera setup successful with latency optimizations!")
    
    def connect_robot(self) -> bool:
        """Connect to xArm7 robot."""
        return self.robot.connect()
    
    def capture_and_preprocess_image(self) -> Tuple[np.ndarray, Image.Image]:
        """
        Capture and preprocess image from camera with latency optimization.
        
        Returns:
            Tuple of (raw_image, preprocessed_image)
        """
        if self.cap is None:
            raise RuntimeError("Camera not initialized")
        
        # Flush old frames from buffer to get the most recent frame
        for _ in range(2):  # Flush 2 frames
            self.cap.grab()
        
        # Get the fresh frame
        ret, frame = self.cap.retrieve()
        if not ret:
            # Fallback to regular read if retrieve fails
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture image from camera")
        
        # Convert BGR to RGB
        raw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        # preprocessed_image = self.preprocessor.preprocess_image(raw_image)
        pil_image = Image.fromarray(raw_image)
        
        return raw_image, pil_image
    
    def predict_action(self, image: Image.Image, instruction: str, 
                      unnorm_key: Optional[str] = None) -> np.ndarray:
        """
        Send image and instruction to OpenVLA server and get predicted action.
        """
        image_array = np.array(image)
        payload = {
            "image": image_array,
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
                    display_images: bool = True, save_episode: bool = True):
        """
        Execute a task with latency monitoring.
        
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
        
        # Start episode if saving is enabled
        episode_id = None
        if save_episode:
            episode_id = self.episode_manager.start_episode(instruction)

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
        
        latency_measurements = []
        
        for step in range(max_steps):
            step_start_time = time.time()
            
            try:
                # # Measure camera capture latency
                # capture_start = time.time()
                raw_image, processed_image = self.capture_and_preprocess_image()
                # capture_time = time.time() - capture_start
                
                # # Measure prediction latency
                # prediction_start = time.time()
                raw_action = self.predict_action(processed_image, instruction, unnorm_key)
                # prediction_time = time.time() - prediction_start
                
                # # Measure postprocessing latency
                # postprocess_start = time.time()
                robot_state = self.robot.get_state()
                absolute_action = self.postprocessor.postprocess_action(raw_action, robot_state, unnorm_key)
                # postprocess_time = time.time() - postprocess_start
                
                # # Store latency measurements
                # total_latency = capture_time + prediction_time + postprocess_time
                # latency_measurements.append({
                #     'capture': capture_time * 1000,  # ms
                #     'prediction': prediction_time * 1000,
                #     'postprocess': postprocess_time * 1000,
                #     'total': total_latency * 1000
                # })
                
                # # Print latency info every 10 steps
                # if step % 10 == 0:
                #     print(f"Step {step+1}: Latency - Capture: {capture_time*1000:.1f}ms, "
                #           f"Prediction: {prediction_time*1000:.1f}ms, "
                #           f"Total: {total_latency*1000:.1f}ms")
                
                # Get current robot state
                # robot_state = self.robot.get_state()
                actual_deltas = absolute_action[:3] - robot_state.position

                # Display image if requested
                stop_episode = False
                end_episode = False
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
                        stop_episode = True
                    elif key == ord('e'):
                        print("Ending episode due to user input...")
                        end_episode = True
                    elif key == ord(' '):  # Space bar for emergency stop
                        print("Emergency stop!")
                        self.safety.emergency_stop = True
                        continue

                if stop_episode:
                    break
                
                if end_episode:
                    break

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
                
                # Save step data if episode saving is enabled
                if save_episode:
                    step_data = {
                        'step': step,
                        'timestamp': time.time(),
                        'positioning_mode': 'absolute' if self.use_absolute else 'delta',
                        # 'raw_image': raw_image,
                        # 'processed_image': np.array(processed_image),
                        'instruction': instruction,
                        'unnorm_key': unnorm_key,
                        'robot_state': {
                            'position': robot_state.position,
                            'rotation': robot_state.rotation,
                            'gripper': robot_state.gripper
                        },
                        'vla_input': {
                            'image': np.array(processed_image),
                            'instruction': instruction,
                            'unnorm_key': unnorm_key
                        },
                        'vla_output': {
                            'raw_action': raw_action,
                            'processed_action': absolute_action
                        },
                        'actual_deltas': actual_deltas,
                        'execution_success': execution_success
                    }
                    self.episode_manager.add_step(step_data)
                
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
        
        # # Print latency statistics
        # if latency_measurements:
        #     avg_capture = np.mean([m['capture'] for m in latency_measurements])
        #     avg_prediction = np.mean([m['prediction'] for m in latency_measurements])
        #     avg_total = np.mean([m['total'] for m in latency_measurements])
            
        #     print(f"\nLatency Statistics:")
        #     print(f"  Average Capture: {avg_capture:.1f}ms")
        #     print(f"  Average Prediction: {avg_prediction:.1f}ms")
        #     print(f"  Average Total: {avg_total:.1f}ms")
        
        # # Save execution data
        
        
        # Disconnect robot
        self.robot.disconnect()
        # End episode and get success label
        episode_success = False
        if save_episode:
            episode_success = self.episode_manager.end_episode(manual_success_input=True)
        
        print(f"Task execution completed!")
        print(f"Successful steps: {successful_steps}/{step+1}")
        print(f"Positioning mode used: {'ABSOLUTE' if self.use_absolute else 'DELTA'}")
        if save_episode:
            print(f"Episode result: {'SUCCESS' if episode_success else 'FAILURE'}")
        
        # Disconnect robot
        self.robot.disconnect()
        
        return episode_success
    
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
            
            print(f"DEBUG - Raw action from server:")
            print(f"  Shape: {raw_action.shape}")
            print(f"  Values: {raw_action}")
            print(f"  Min/Max: [{raw_action.min():.3f}, {raw_action.max():.3f}]")
            print(f"")
            print(f"Positioning Mode: {'ABSOLUTE' if self.use_absolute else 'DELTA'}")
            print(f"Raw action: {raw_action}")
            print(f"Processed action: {absolute_action}")
            print(f"Action breakdown:")
            
            if self.use_absolute:
                print(f"  Raw Position (absolute): [{raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}]")
                print(f"  Scaled Position (absolute): [{absolute_action[0]:.3f}, {absolute_action[1]:.3f}, {absolute_action[2]:.3f}] mm")
                print(f"  Movement Delta: [{actual_deltas[0]:.3f}, {actual_deltas[1]:.3f}, {actual_deltas[2]:.3f}] mm")
            else:
                print(f"  Position Delta (raw): [{raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}]")
                print(f"  Position Delta (scaled): [{actual_deltas[0]:.3f}, {actual_deltas[1]:.3f}, {actual_deltas[2]:.3f}] mm")
            
            print(f"  Rotation: [{raw_action[3]:.3f}, {raw_action[4]:.3f}, {raw_action[5]:.3f}]")
            print(f"  Current Position: [{robot_state.position[0]:.1f}, {robot_state.position[1]:.1f}, {robot_state.position[2]:.1f}] mm")
            print(f"  Target Position: [{absolute_action[0]:.1f}, {absolute_action[1]:.1f}, {absolute_action[2]:.1f}] mm")
            print(f"  Target Rotation: [{absolute_action[3]:.1f}, {absolute_action[4]:.1f}, {absolute_action[5]:.1f}] deg")
            print(f"  Gripper: {absolute_action[6]:.3f}")
            
            # Display image with action overlay
            display_img = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            cv2.putText(display_img, f"Instruction: {instruction}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            mode_text = "ABSOLUTE" if self.use_absolute else "DELTA"
            cv2.putText(display_img, f"Mode: {mode_text}", (10, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            if self.use_absolute:
                cv2.putText(display_img, f"Target: {absolute_action[:3]}", (10, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(display_img, f"Delta: {actual_deltas}", (10, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                cv2.putText(display_img, f"Delta: {actual_deltas}", (10, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(display_img, f"Target: {absolute_action[:3]}", (10, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            cv2.imshow("Single Prediction Test - xArm7", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during prediction: {e}")

    def cleanup(self):
        """Release all resources."""
        if self.use_threaded_camera and self.threaded_camera:
            self.threaded_camera.stop()
        elif self.cap is not None:
            self.cap.release()
        
        self.robot.disconnect()
        cv2.destroyAllWindows()
        print("All resources released.")


def main():
    parser = argparse.ArgumentParser(description="OpenVLA xArm7 Robot Control Client")
    parser.add_argument("--server_url", type=str, default="http://10.136.109.82:8000",
                       help="URL of the OpenVLA server")
    parser.add_argument("--camera_index", type=int, default=6,
                       help="Camera index to use for image capture")
    parser.add_argument("--robot_ip", type=str, default="192.168.1.239",
                       help="IP address of xArm7 robot")
    parser.add_argument("--instruction", type=str, required=True,
                       help="Task instruction for the robot")
    parser.add_argument("--unnorm_key", type=str, default="ucsd_pick_and_place_dataset_converted_externally_to_rlds",
                       help="Dataset key for action denormalization")
    parser.add_argument("--mode", type=str, choices=["single", "execute"], default="execute",
                       help="Mode: 'single' for prediction test, 'execute' for full task execution")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum number of steps for task execution")
    parser.add_argument("--control_frequency", type=float, default=5.0,
                       help="Control frequency in Hz")
    parser.add_argument("--no_display", action="store_true",
                       help="Disable image display")
    parser.add_argument("--save_image", action="store_true",
                       help="Save captured images")
    parser.add_argument("--no_episode_save", action="store_true",
                       help="Disable episode data saving")
    parser.add_argument("--episode_dir", type=str, default="episode_data",
                       help="Directory to save episode data")
    parser.add_argument("--use_threaded_camera", action="store_true",
                       help="Use threaded camera capture for low latency")
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
        use_absolute=use_absolute,
        use_threaded_camera=args.use_threaded_camera,
        episode_save_dir=args.episode_dir
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
                save_episode=not args.no_episode_save
            )
    
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()