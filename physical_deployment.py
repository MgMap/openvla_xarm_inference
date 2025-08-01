import requests
import json_numpy
json_numpy.patch()
import numpy as np
import cv2
from xarm.wrapper import XArmAPI
import time
import tensorflow as tf

class XArmVLAController:
    def __init__(self, xarm_ip="192.168.1.239", vla_server_url="http://10.136.109.82:8000"):
        # Initialize XArm7
        self.arm = XArmAPI(xarm_ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position mode
        self.arm.set_state(state=0)  # Sport state
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        # VLA server URL
        self.vla_url = vla_server_url + "/act"
        
        # Camera setup (adjust for your camera)
        self.camera = cv2.VideoCapture(6)  # or your camera index
    
    def resize_image_for_openvla(self, img, resize_size=(224, 224)):
        """
        Resize image following the same process as OpenVLA training data.
        Based on bridgev2_utils.resize_image()
        """
        # Convert numpy array to tensor
        img_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
        
        # Encode as JPEG, as done in RLDS dataset builder
        img_encoded = tf.image.encode_jpeg(img_tensor)
        
        # Immediately decode back
        img_decoded = tf.io.decode_image(img_encoded, expand_animations=False, dtype=tf.uint8)
        
        # Resize using lanczos3 with antialias (matches training)
        img_resized = tf.image.resize(img_decoded, resize_size, method="lanczos3", antialias=True)
        
        # Clip and cast back to uint8
        img_final = tf.cast(tf.clip_by_value(tf.round(img_resized), 0, 255), tf.uint8)
        
        return img_final.numpy()
    
    def get_camera_image(self):
        """Capture image from camera with proper preprocessing"""
        ret, frame = self.camera.read()
        if ret:
            # Resize to expected input size (256x256)
            frame = cv2.resize(frame, (256, 256))
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure the image is uint8 and in range [0, 255]
            frame = frame.astype(np.uint8)
            
            # Convert to float32 and normalize to [0, 1] range
            frame = frame.astype(np.float32) / 255.0
            
            return frame
        return None
    
    def get_camera_image_with_debug(self):
        """Capture image with debugging info - properly formatted for OpenVLA"""
        # Flush camera buffer
        for _ in range(3):
            ret, frame = self.camera.read()
            if not ret:
                return None
        
        # Capture actual frame
        ret, frame = self.camera.read()
        if ret:
            # Convert BGR to RGB FIRST (before any resizing)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize using OpenVLA's method to match training distribution
            frame_resized = self.resize_image_for_openvla(frame_rgb, (224, 224))
            
            # DON'T normalize to [0,1] - keep as uint8 [0,255]
            # The model/processor handles normalization internally
            
            # Calculate image signature for debugging
            import hashlib
            signature = hashlib.md5(frame_resized.tobytes()).hexdigest()[:8]
            print(f"Image signature: {signature}")
            print(f"Image shape: {frame_resized.shape}, dtype: {frame_resized.dtype}")
            print(f"Image value range: {frame_resized.min()} to {frame_resized.max()}")
            
            # Store for comparison
            if hasattr(self, 'last_signature'):
                if signature == self.last_signature:
                    print("⚠️  WARNING: Same image as previous step!")
                else:
                    print("✅ New image captured")
            
            self.last_signature = signature
            return frame_resized  # Return uint8 image, NOT normalized float32
        return None
    
    def predict_action(self, instruction, unnorm_key="bridge_orig"):  # Changed default
        """Get action prediction from VLA server"""
        print("📷 Capturing fresh camera image...")
        image = self.get_camera_image_with_debug()
        if image is None:
            print("❌ Failed to capture image")
            return None
        
        try:
            # DON'T convert to list - send numpy array directly
            # json_numpy.patch() handles the serialization
            payload = {
                "image": image,  # Send numpy array directly, not image.tolist()
                "instruction": instruction,
                "unnorm_key": unnorm_key
            }
            
            print(f"📤 Sending request with instruction: '{instruction}' and unnorm_key: '{unnorm_key}'")
            
            response = requests.post(
                self.vla_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60  # Increased timeout
            )
            
            print(f"📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, str) and result == "error":
                    print("❌ Server returned error - check server logs")
                    return None
                print(f"✅ Action received from server")
                return result
            else:
                print(f"❌ HTTP error: {response.status_code}")
                print(f"Response text: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            return None
        except Exception as e:
            print(f"❌ Error predicting action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def execute_action(self, action):
        """Execute predicted action on XArm7"""
        if action is None:
            return False
            
        try:
            # Assuming action is [x, y, z, rx, ry, rz, gripper]
            if len(action) >= 6:
                target_x, target_y, target_z, target_rx, target_ry, target_rz = action[:6]
                
                print(f"Raw action: x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}, rx={target_rx:.2f}, ry={target_ry:.2f}, rz={target_rz:.2f}")
                
                # Get current robot position
                current_pos = self.arm.get_position()[1]  # Returns [ret, [x,y,z,rx,ry,rz]]
                current_x, current_y, current_z, current_rx, current_ry, current_rz = current_pos
                
                print(f"Current position: x={current_x:.2f}, y={current_y:.2f}, z={current_z:.2f}, rx={current_rx:.2f}, ry={current_ry:.2f}, rz={current_rz:.2f}")
                
                # 🔒 SAFETY: Limit Z position to prevent collision
                if target_z < -11.9:
                    print(f"⚠️  Z action {target_z:.2f} would exceed limit -11.9")
                    print(f"   Clamping Z to -11.9")
                    target_z = -11.9

                # 🔍 DEBUG: Check if orientation values are reasonable
                print(f"Orientation change: rx({current_rx:.1f}→{target_rx:.1f}), ry({current_ry:.1f}→{target_ry:.1f}), rz({current_rz:.1f}→{target_rz:.1f})")
                
                # 🔒 SAFETY: Limit orientation changes if they're too large
                max_orientation_change = 45.0  # degrees
                
                def clamp_orientation_change(current, target, max_change):
                    diff = target - current
                    # Handle angle wrapping (e.g., -179 to +179)
                    if abs(diff) > 180:
                        if diff > 0:
                            diff -= 360
                        else:
                            diff += 360
                    
                    if abs(diff) > max_change:
                        clamped = current + np.sign(diff) * max_change
                        print(f"   Clamping orientation change from {diff:.1f}° to {np.sign(diff) * max_change:.1f}°")
                        return clamped
                    return target
                
                # Apply orientation safety limits
                safe_rx = clamp_orientation_change(current_rx, target_rx, max_orientation_change)
                safe_ry = clamp_orientation_change(current_ry, target_ry, max_orientation_change)
                safe_rz = clamp_orientation_change(current_rz, target_rz, max_orientation_change)
                
                print(f"Safe orientation: rx={safe_rx:.2f}, ry={safe_ry:.2f}, rz={safe_rz:.2f}")

                # Move arm with safety limits - using actions directly
                ret = self.arm.set_position(
                    x=target_x,
                    y=target_y,
                    z=target_z,
                    roll=safe_rx,  # Use safe orientation values
                    pitch=safe_ry,
                    yaw=safe_rz,
                    speed=30,  # Slow speed for safety
                    mvacc=50,  # Low acceleration
                    wait=True
                )
                
                # 🔍 DEBUG: Verify the position was actually set
                time.sleep(0.5)  # Give robot time to move
                new_pos = self.arm.get_position()[1]
                new_x, new_y, new_z, new_rx, new_ry, new_rz = new_pos
                print(f"Actual new position: x={new_x:.2f}, y={new_y:.2f}, z={new_z:.2f}, rx={new_rx:.2f}, ry={new_ry:.2f}, rz={new_rz:.2f}")
                
                # Check if orientation actually changed
                orientation_changed = (abs(new_rx - current_rx) > 1.0 or 
                                     abs(new_ry - current_ry) > 1.0 or 
                                     abs(new_rz - current_rz) > 1.0)
                
                if not orientation_changed:
                    print("⚠️  WARNING: Robot orientation did not change! This might be:")
                    print("   1. Robot is in position mode but orientation is locked")
                    print("   2. Orientation values are in wrong units (radians vs degrees)")
                    print("   3. Robot has orientation limits enabled")
                    
                    # Try alternative: Set servo angle mode for orientation
                    print("   Trying alternative orientation control...")
                    # You might need to use set_servo_angle() instead for orientation
                
                # Handle gripper if action includes it
                if len(action) > 6:
                    gripper_action = action[6]
                    print(f"Gripper action: {gripper_action:.3f}")
                    if gripper_action > 0.5:  # Open gripper
                        self.arm.set_gripper_position(800, wait=True)
                        print("Opening gripper")
                    else:  # Close gripper
                        self.arm.set_gripper_position(0, wait=True)
                        print("Closing gripper")
                
                if ret == 0:
                    print("✓ Action executed successfully")
                else:
                    print(f"✗ Action execution failed with code: {ret}")
                    # Print XArm error code meaning
                    error_meanings = {
                        1: "Emergency stop",
                        2: "Not enabled",
                        3: "Not ready",
                        4: "Mode error",
                        5: "Limit",
                        6: "Joint limit",
                        7: "Speed limit",
                        8: "Acceleration limit"
                    }
                    if ret in error_meanings:
                        print(f"   Error meaning: {error_meanings[ret]}")
                
                return ret == 0
            return False
        except Exception as e:
            print(f"Error executing action: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_command(self, instruction, unnorm_key="bridge_orig"):
        """Main function to run a command"""
        print(f"Executing instruction: {instruction}")
        
        # Get action prediction
        action = self.predict_action(instruction, unnorm_key)
        if action is None:
            print("Failed to get action prediction")
            return False
        
        print(f"Predicted action: {action}")
        
        # Execute action
        success = self.execute_action(action)
        if success:
            print("Action executed successfully")
        else:
            print("Failed to execute action")
        
        return success
    
    def run_multi_step_command(self, instruction, unnorm_key="bridge_orig", max_steps=50, step_delay=0.5):
        """Execute a command with multiple action steps"""
        print(f"Starting multi-step execution: {instruction}")
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            
            # Get action prediction from current observation
            action = self.predict_action(instruction, unnorm_key)
            if action is None:
                print("Failed to get action prediction")
                return False
            
            print(f"Predicted action: {action}")
            
            # Execute action
            success = self.execute_action(action)
            if not success:
                print("Failed to execute action")
                return False
            
            # Wait before next step
            time.sleep(step_delay)
            
            # Optional: Add termination condition
            # if self.is_task_complete():
            #     print("Task completed!")
            #     return True
        
        print("Reached maximum steps")
        return True
    
    def run_task_sequence(self, task_sequence, unnorm_key="bridge_orig"):
        """Execute a sequence of related commands"""
        print(f"Starting task sequence with {len(task_sequence)} commands")
        
        for i, instruction in enumerate(task_sequence):
            print(f"\n=== Task {i+1}/{len(task_sequence)}: {instruction} ===")
            
            # Execute each instruction as a multi-step command
            success = self.run_multi_step_command(instruction, unnorm_key, max_steps=20)
            if not success:
                print(f"Failed at task {i+1}: {instruction}")
                return False
            
            print(f"✓ Completed: {instruction}")
        
        print("All tasks completed successfully!")
        return True
    
    def run_visual_servoing(self, instruction, unnorm_key="bridge_orig", max_steps=100):
        """Continuous visual servoing until task completion"""
        print(f"Starting visual servoing: {instruction}")
        
        for step in range(max_steps):
            print(f"Step {step + 1}: Getting current observation...")
            
            # 🔥 CRITICAL: Get FRESH camera image after previous action
            action = self.predict_action(instruction, unnorm_key)
            if action is None:
                print("No action prediction - stopping")
                break
            
            print(f"Action: {action}")
            
            # Execute action (robot moves)
            success = self.execute_action(action)
            if not success:
                print("Action execution failed")
                break
            
            # 🔥 IMPORTANT: Wait for robot to finish moving before next image
            time.sleep(1.0)  # Increase this delay
            
            # Optional: Save image to verify it's changing
            if step < 5:  # Save first few images for debugging
                self.save_camera_image(f"step_{step+1}_after_action.jpg")
        
        print(f"Visual servoing completed after {step + 1} steps")
        return True
    
    def cleanup(self):
        """Clean up resources"""
        self.camera.release()
        self.arm.disconnect()
    
    def run_interactive_continuous(self):
        """Interactive mode with continuous execution"""
        print("Interactive continuous mode. Commands:")
        print("  - Enter instruction to start continuous execution")
        print("  - 'stop' to stop current execution")
        print("  - 'quit' to exit")
        
        while True:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.lower() == 'stop':
                print("Stopping current execution...")
                continue
            elif command:
                print(f"Starting continuous execution: {command}")
                self.run_visual_servoing(command)
    
    def test_camera(self):
        """Test camera and show what it sees"""
        print("Testing camera...")
        
        # Check if camera is working
        image = self.get_camera_image()
        if image is None:
            print("❌ Camera not working!")
            return False
        
        print(f"✅ Camera working! Image shape: {image.shape}")
        
        # Save a test image
        self.save_camera_image("test_camera_view.jpg")
        
        # Option: Show live feed
        show_feed = input("Show live camera feed? (y/n): ").lower().strip()
        if show_feed == 'y':
            self.show_camera_feed(5)  # Show for 5 seconds
        
        return True
    
    def save_camera_image(self, filename="current_view.jpg"):
        """Save current camera image to file"""
        image = self.get_camera_image_with_debug()
        if image is not None:
            # Image is already uint8, just convert RGB to BGR for saving
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, image_bgr)
            print(f"Camera image saved as {filename}")
            return True
        else:
            print("Failed to capture camera image")
            return False
    
    def show_camera_feed(self, duration=10):
        """Show live camera feed for specified duration (seconds)"""
        print(f"Showing camera feed for {duration} seconds. Press 'q' to quit early.")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = self.camera.read()
            if ret:
                # Show original frame
                cv2.imshow('Camera Feed', frame)
                
                # Also show what gets sent to VLA (resized)
                frame_resized = cv2.resize(frame, (256, 256))
                cv2.imshow('VLA Input (256x256)', frame_resized)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to read from camera")
                break
        
        cv2.destroyAllWindows()
    
    def check_robot_config(self):
        """Check robot configuration and capabilities"""
        print("🔍 Robot Configuration Check:")
        
        # Check current mode
        mode = self.arm.get_mode()
        print(f"Current mode: {mode}")
        
        # Check state
        state = self.arm.get_state()
        print(f"Current state: {state}")
        
        # Check if motion is enabled
        print(f"Motion enabled: {self.arm.get_is_moving()}")
        
        # Check position
        pos = self.arm.get_position()
        print(f"Current position: {pos}")
        
        # Check if there are any limits or constraints
        joint_limits = self.arm.get_joint_limit_toggle()
        print(f"Joint limits enabled: {joint_limits}")
        
        # Check servo angle mode capabilities
        servo_angle = self.arm.get_servo_angle()
        print(f"Servo angles: {servo_angle}")
        
        return True

# Usage example
if __name__ == "__main__":
    controller = XArmVLAController(
        xarm_ip="192.168.1.239",
        vla_server_url="http://10.136.109.82:8000"
    )
    
    try:
        # Test camera first
        if not controller.test_camera():
            print("Camera test failed. Exiting.")
            exit(1)
        
        # Save an image before starting
        controller.save_camera_image("before_task.jpg")
        
        # Run the task
        controller.run_visual_servoing("move towards the orange cube", unnorm_key='ucsd_kitchen_dataset_converted_externally_to_rlds', max_steps=50)

        # Save an image after task
        controller.save_camera_image("after_task.jpg")

    finally:
        controller.cleanup()