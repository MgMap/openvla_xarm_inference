import numpy as np
import torch
import time
from pynput import keyboard
from xarm.wrapper import XArmAPI

class KeyboardController:
    """Handles keyboard input for modifying a given target variable (e.g., wrist position)."""
    
    def __init__(self, arm, target_pos, speed=50):
        """
        :param target_pos: Reference to the target wrist position tensor.
        :param speed: Step size for position updates.
        """
        self.arm = arm
        self.target_pos = target_pos  # torch tensor [x, y, z]
        self.speed = speed  # Movement speed
        self.running = True

        # Start a background thread for keyboard input
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        """Handle key presses and update the target position."""
        try:
            if key.char == "w":  # Move UP (Z+)
                self.target_pos[2] += self.speed
            elif key.char == "s":  # Move DOWN (Z-)
                self.target_pos[2] -= self.speed
            elif key.char == "a":  # Move LEFT (X+)
                self.target_pos[0] += self.speed
            elif key.char == "d":  # Move RIGHT (X-)
                self.target_pos[0] -= self.speed
            elif key.char == "q":  # Move FORWARD (Y+)
                self.target_pos[1] += self.speed
            elif key.char == "e":  # Move BACKWARD (Y-)
                self.target_pos[1] -= self.speed
            elif key.char == "r":
                # Reset to home position
                self.target_pos[:] = torch.tensor(self.arm.position[:3])
        except AttributeError:
            pass  # Ignore special keys (Shift, Ctrl, etc.)

    def stop(self):
        """Stop the keyboard listener."""
        self.running = False
        self.listener.stop()

if __name__ == "__main__":
    # Connect to xArm
    arm = XArmAPI('192.168.1.239')  # <-- Replace with your xArm's IP
    arm.motion_enable(enable=True)
    arm.set_mode(7)
    arm.set_state(0)
    time.sleep(0.5)

    # Get current position [x, y, z, roll, pitch, yaw]
    pos = arm.position
    if pos is None:
        print("Failed to get xArm position!")
        exit(1)
    target_pos = torch.tensor(pos[:3], dtype=torch.float32)

    controller = KeyboardController(arm, target_pos)
    print("Use WASDQE to move the end-effector in XYZ. Press Ctrl+C to exit.")

    try:
        while controller.running:
            # Send new position (keep orientation fixed)
            current_ori = arm.position[3:]  # roll, pitch, yaw
            xyz = target_pos.tolist()
            arm.set_position(*xyz, *current_ori, wait=False, speed=50, mvacc=5000)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        controller.stop()
        arm.disconnect()