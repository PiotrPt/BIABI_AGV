"""Robot Agent - Physical robot with differential drive kinetics."""

import math
import numpy as np


class RobotAgent:
    """
    A differential-drive robot with 2 independent wheels and 3 distance sensors.
    """

    def __init__(self, start_x: float, start_y: float, start_angle: float = 0.0, 
                 width: float = 30, height: float = 20, max_speed: float = 1.0):
        """
        Initialize robot at given position.
        
        Args:
            start_x: Initial X position (pixels)
            start_y: Initial Y position (pixels)
            start_angle: Initial angle in degrees (0 = facing right)
            width: Robot body width (pixels)
            height: Robot body height (pixels)
            max_speed: Maximum motor speed (1.0 = full speed)
        """
        self.x = float(start_x)
        self.y = float(start_y)
        self.angle_rad = math.radians(float(start_angle))  # Store as radians
        
        self.width = width
        self.height = height
        self.max_speed = max_speed
        
        # Motor speeds (-1.0 to 1.0)
        self.left_speed = 0.0
        self.right_speed = 0.0
        
        # Robot dynamics - axle distance (wheelbase)
        self.axle_distance = 25.0  # Distance between wheels (pixels)
        
        # Track history for reward calculation
        self.prev_x = self.x
        self.prev_y = self.y
        self.total_distance = 0.0
        
    def set_motors(self, left: float, right: float) -> None:
        """
        Set motor speeds.
        
        Args:
            left: Left motor speed (-1.0 to 1.0)
            right: Right motor speed (-1.0 to 1.0)
        """
        self.left_speed = np.clip(float(left), -1.0, 1.0)
        self.right_speed = np.clip(float(right), -1.0, 1.0)
    
    def update(self, dt: float = 0.016) -> None:
        """
        Update robot position using differential drive kinematics.
        
        For a differential drive robot:
        - Linear velocity = (v_right + v_left) / 2
        - Angular velocity = (v_right - v_left) / axle_distance
        
        Args:
            dt: Time delta in seconds
        """
        # Convert motor speeds to velocities (pixels/second)
        # Max speed ~ 300 pixels/second
        max_linear_speed = 300.0
        
        v_right = self.right_speed * max_linear_speed
        v_left = self.left_speed * max_linear_speed
        
        # Differential drive kinematics
        v_linear = (v_right + v_left) / 2.0
        v_angular = (v_right - v_left) / self.axle_distance
        
        # Update position (Euler integration)
        if abs(v_linear) > 1e-6:  # Avoid division by zero
            # Forward movement with rotation
            self.x += v_linear * math.cos(self.angle_rad) * dt
            self.y += v_linear * math.sin(self.angle_rad) * dt
            
            # Track distance traveled
            distance_moved = abs(v_linear) * dt
            self.total_distance += distance_moved
        
        # Update angle
        self.angle_rad += v_angular * dt
        
        # Normalize angle to [-π, π]
        while self.angle_rad > math.pi:
            self.angle_rad -= 2 * math.pi
        while self.angle_rad < -math.pi:
            self.angle_rad += 2 * math.pi
    
    def get_center(self) -> tuple:
        """Get robot center (x, y)."""
        return (self.x, self.y)
    
    def get_corners(self) -> list:
        """
        Get robot body corners for collision detection.
        Robot is axis-aligned rectangle rotated around center.
        
        Returns:
            List of 4 (x, y) corners
        """
        cos_a = math.cos(self.angle_rad)
        sin_a = math.sin(self.angle_rad)
        
        # Half dimensions
        hw = self.width / 2.0
        hh = self.height / 2.0
        
        # Corners in local coordinates
        local_corners = [
            (-hw, -hh),  # Bottom-left
            (hw, -hh),   # Bottom-right
            (hw, hh),    # Top-right
            (-hw, hh),   # Top-left
        ]
        
        # Rotate and translate to world coordinates
        global_corners = []
        for lx, ly in local_corners:
            gx = self.x + lx * cos_a - ly * sin_a
            gy = self.y + lx * sin_a + ly * cos_a
            global_corners.append((gx, gy))
        
        return global_corners
    
    def get_bounding_circle(self) -> tuple:
        """
        Get robot bounding circle for quick collision checks.
        
        Returns:
            (center_x, center_y, radius)
        """
        radius = math.sqrt(self.width**2 + self.height**2) / 2.0
        return (self.x, self.y, radius)
    
    def check_collision_with_line(self, p1: tuple, p2: tuple, line_width: float = 15.0) -> bool:
        """
        Check collision between robot and line segment (obstacle).
        
        Args:
            p1: Line start point (x, y)
            p2: Line end point (x, y)
            line_width: Thickness of line (pixels)
        
        Returns:
            True if collision detected
        """
        # Use bounding circle for quick rejection
        cx, cy, r = self.get_bounding_circle()
        
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Vector from p1 to circle center
        fx = cx - p1[0]
        fy = cy - p1[1]
        
        # Length squared of line segment
        len_sq = dx*dx + dy*dy
        
        if len_sq < 1e-6:  # p1 and p2 are same point
            dist = math.sqrt(fx*fx + fy*fy)
            return dist < (r + line_width / 2.0)
        
        # Parameter t of closest point on line segment
        t = max(0.0, min(1.0, (fx*dx + fy*dy) / len_sq))
        
        # Closest point on line segment
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        
        # Distance from circle center to closest point
        dist_x = cx - closest_x
        dist_y = cy - closest_y
        dist = math.sqrt(dist_x*dist_x + dist_y*dist_y)
        
        # Collision if distance < robot radius + line width
        return dist < (r + line_width / 2.0)
    
    def check_collision_with_obstacle(self, obstacle: dict) -> bool:
        """
        Check collision with obstacle (line obstacle from JSON).
        
        Args:
            obstacle: Obstacle dict with 'points' and 'width'
        
        Returns:
            True if collision detected
        """
        points = obstacle.get('points', [])
        width = obstacle.get('width', 15.0)
        
        if len(points) < 2:
            return False
        
        # Check collision with each line segment
        for i in range(len(points) - 1):
            if self.check_collision_with_line(points[i], points[i + 1], width):
                return True
        
        return False
    
    def check_collision_with_obstacles(self, obstacles: list) -> bool:
        """
        Check collision with all obstacles.
        
        Args:
            obstacles: List of obstacle dicts
        
        Returns:
            True if any collision detected
        """
        for obstacle in obstacles:
            if self.check_collision_with_obstacle(obstacle):
                return True
        return False
    
    def get_average_speed(self) -> float:
        """
        Get average motor speed (0 to 1).
        
        Returns:
            Average absolute speed of both motors
        """
        return (abs(self.left_speed) + abs(self.right_speed)) / 2.0
    
    def is_stuck(self, threshold: float = 0.05) -> bool:
        """
        Check if robot is not moving (stuck).
        
        Args:
            threshold: Speed threshold below which robot is considered stuck
        
        Returns:
            True if average speed is below threshold (actual movement, not just motor commands)
        """
        avg_speed = self.get_average_speed()
        return avg_speed < threshold
    
    def reset(self, start_x: float, start_y: float, start_angle: float = 0.0) -> None:
        """Reset robot to starting position."""
        self.x = float(start_x)
        self.y = float(start_y)
        self.angle_rad = math.radians(float(start_angle))
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.total_distance = 0.0
