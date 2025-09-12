import time
import numpy as np
from typing import List, Tuple, Optional

# Hardware imports - only import if available
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("Hardware libraries not available - running in simulation mode")

class RobotArmController:
    """
    Controls robot arm servos through PCA9685 PWM controller
    """
    
    def __init__(self, num_servos: int = 4, min_pulse: int = 500, max_pulse: int = 2500):
        """
        Initialize the robot arm controller
        
        Args:
            num_servos: Number of servo motors in the arm
            min_pulse: Minimum pulse width for servo (microseconds)
            max_pulse: Maximum pulse width for servo (microseconds)
        """
        self.num_servos = num_servos
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        
        # Initialize I2C bus and PCA9685
        if HARDWARE_AVAILABLE:
            try:
                self.i2c = busio.I2C(board.SCL, board.SDA)
                self.pca = PCA9685(self.i2c)
                self.pca.frequency = 50  # 50Hz for servo control
                print("PCA9685 initialized successfully")
            except Exception as e:
                print(f"Error initializing PCA9685: {e}")
                self.pca = None
        else:
            print("Hardware not available - simulating servo control")
            self.pca = None
        
        # Current servo positions (in degrees, 0-180)
        self.current_positions = np.zeros(self.num_servos)
        
        # Servo angle limits (degrees)
        self.angle_limits = [(0, 180) for _ in range(self.num_servos)]
        
        # Initialize servos to neutral position
        self.reset_to_neutral()
    
    def angle_to_pulse(self, angle: float) -> int:
        """Convert servo angle (0-180 degrees) to pulse width"""
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        
        # Linear interpolation between min and max pulse
        pulse = self.min_pulse + (angle / 180.0) * (self.max_pulse - self.min_pulse)
        return int(pulse)
    
    def set_servo_angle(self, servo_id: int, angle: float) -> bool:
        """
        Set specific servo to given angle
        
        Args:
            servo_id: Servo channel (0-15 on PCA9685)
            angle: Target angle in degrees (0-180)
            
        Returns:
            bool: Success status
        """
        if self.pca is None:
            print("PCA9685 not initialized")
            return False
        
        if servo_id < 0 or servo_id >= self.num_servos:
            print(f"Invalid servo ID: {servo_id}")
            return False
        
        # Clamp angle to limits
        min_angle, max_angle = self.angle_limits[servo_id]
        angle = max(min_angle, min(max_angle, angle))
        
        try:
            pulse = self.angle_to_pulse(angle)
            self.pca.channels[servo_id].duty_cycle = int(pulse * 65535 / 20000)  # Convert to 16-bit value
            self.current_positions[servo_id] = angle
            return True
        except Exception as e:
            print(f"Error setting servo {servo_id} to angle {angle}: {e}")
            return False
    
    def set_all_servos(self, angles: List[float]) -> bool:
        """
        Set all servos to specified angles
        
        Args:
            angles: List of angles for each servo
            
        Returns:
            bool: Success status
        """
        if len(angles) != self.num_servos:
            print(f"Expected {self.num_servos} angles, got {len(angles)}")
            return False
        
        success = True
        for i, angle in enumerate(angles):
            if not self.set_servo_angle(i, angle):
                success = False
        
        return success
    
    def get_current_positions(self) -> np.ndarray:
        """Get current servo positions"""
        return self.current_positions.copy()
    
    def reset_to_neutral(self):
        """Reset all servos to neutral position (90 degrees)"""
        neutral_angles = [90.0] * self.num_servos
        self.set_all_servos(neutral_angles)
        time.sleep(1)  # Allow time for servos to move
    
    def move_servo_smoothly(self, servo_id: int, target_angle: float, steps: int = 20, delay: float = 0.05):
        """
        Move servo smoothly to target angle
        
        Args:
            servo_id: Servo to move
            target_angle: Target angle
            steps: Number of intermediate steps
            delay: Delay between steps (seconds)
        """
        if servo_id < 0 or servo_id >= self.num_servos:
            return
        
        current_angle = self.current_positions[servo_id]
        angle_diff = target_angle - current_angle
        
        for i in range(steps + 1):
            intermediate_angle = current_angle + (angle_diff * i / steps)
            self.set_servo_angle(servo_id, intermediate_angle)
            time.sleep(delay)
    
    def set_angle_limits(self, servo_id: int, min_angle: float, max_angle: float):
        """Set angle limits for a specific servo"""
        if 0 <= servo_id < self.num_servos:
            self.angle_limits[servo_id] = (min_angle, max_angle)
    
    def cleanup(self):
        """Cleanup PCA9685 resources"""
        if self.pca is not None:
            self.pca.deinit()

# Manual control interface
def manual_control_interface():
    """Interactive interface for manual servo control"""
    controller = RobotArmController()
    
    print("\n=== Robot Arm Manual Control ===")
    print("Commands:")
    print("  set <servo_id> <angle>  - Set servo angle (0-180 degrees)")
    print("  smooth <servo_id> <angle> - Move servo smoothly")
    print("  all <angle1> <angle2> ... - Set all servos")
    print("  status                  - Show current positions")
    print("  reset                   - Reset to neutral position")
    print("  limits <servo_id> <min> <max> - Set angle limits")
    print("  quit                    - Exit")
    print()
    
    try:
        while True:
            command = input("Enter command: ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == "quit":
                break
            elif cmd == "set" and len(command) == 3:
                try:
                    servo_id = int(command[1])
                    angle = float(command[2])
                    if controller.set_servo_angle(servo_id, angle):
                        print(f"Servo {servo_id} set to {angle}°")
                    else:
                        print("Failed to set servo")
                except ValueError:
                    print("Invalid servo ID or angle")
            
            elif cmd == "smooth" and len(command) == 3:
                try:
                    servo_id = int(command[1])
                    angle = float(command[2])
                    controller.move_servo_smoothly(servo_id, angle)
                    print(f"Servo {servo_id} moved smoothly to {angle}°")
                except ValueError:
                    print("Invalid servo ID or angle")
            
            elif cmd == "all" and len(command) == controller.num_servos + 1:
                try:
                    angles = [float(x) for x in command[1:]]
                    if controller.set_all_servos(angles):
                        print(f"All servos set to: {angles}")
                    else:
                        print("Failed to set servos")
                except ValueError:
                    print("Invalid angles")
            
            elif cmd == "status":
                positions = controller.get_current_positions()
                print("Current positions:")
                for i, pos in enumerate(positions):
                    print(f"  Servo {i}: {pos:.1f}°")
            
            elif cmd == "reset":
                controller.reset_to_neutral()
                print("Reset to neutral position")
            
            elif cmd == "limits" and len(command) == 4:
                try:
                    servo_id = int(command[1])
                    min_angle = float(command[2])
                    max_angle = float(command[3])
                    controller.set_angle_limits(servo_id, min_angle, max_angle)
                    print(f"Servo {servo_id} limits set to {min_angle}°-{max_angle}°")
                except ValueError:
                    print("Invalid values")
            
            else:
                print("Invalid command or wrong number of arguments")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        controller.cleanup()

if __name__ == "__main__":
    manual_control_interface()
