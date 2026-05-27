"""Sensors - Distance sensors using raycasting."""

import math
import numpy as np


class Sensors:
    """
    3 distance sensors using raycasting from robot.
    Angles relative to robot orientation.
    """
    
    def __init__(self, sensor_angles: list = None, max_distance: float = 200.0):
        """
        Initialize sensors.
        
        Args:
            sensor_angles: List of sensor angles in degrees relative to robot (default: [-60, 0, 60])
            max_distance: Maximum detection distance (pixels)
        """
        self.sensor_angles = sensor_angles or [-60, 0, 60]
        self.max_distance = max_distance
        
        # Convert to radians
        self.sensor_angles_rad = [math.radians(angle) for angle in self.sensor_angles]
        
        # Store last readings
        self.readings = [1.0] * len(self.sensor_angles)  # Normalized [0, 1]
    
    def raycast_to_obstacle(self, start_x: float, start_y: float, angle_rad: float, 
                           obstacles: list) -> float:
        """
        Cast ray from robot position and find distance to first obstacle.
        
        Args:
            start_x: Ray start X (robot center)
            start_y: Ray start Y (robot center)
            angle_rad: Ray direction in radians
            obstacles: List of obstacle dicts
        
        Returns:
            Normalized distance [0, 1] where 1 = max_distance (no collision)
        """
        # Ray direction
        ray_dx = math.cos(angle_rad)
        ray_dy = math.sin(angle_rad)
        
        min_distance = self.max_distance
        
        # Check ray collision with each obstacle
        for obstacle in obstacles:
            points = obstacle.get('points', [])
            width = obstacle.get('width', 15.0)
            
            if len(points) < 2:
                continue
            
            # Check ray intersection with each line segment
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                # Check ray-line intersection
                dist = self._ray_line_intersection(
                    start_x, start_y, ray_dx, ray_dy,
                    p1[0], p1[1], p2[0], p2[1],
                    width / 2.0
                )
                
                if dist >= 0 and dist < min_distance:
                    min_distance = dist
        
        # Normalize distance [0, 1]
        # 0 = obstacle very close, 1 = no obstacle (at max distance)
        normalized = 1.0 - (min_distance / self.max_distance)
        return max(0.0, min(1.0, normalized))
    
    def _ray_line_intersection(self, ray_x: float, ray_y: float, ray_dx: float, ray_dy: float,
                              line_x1: float, line_y1: float, line_x2: float, line_y2: float,
                              line_radius: float) -> float:
        """
        Calculate distance from ray start to closest point on line segment.
        
        Args:
            ray_x, ray_y: Ray start point
            ray_dx, ray_dy: Ray direction (normalized)
            line_x1, line_y1: Line segment start
            line_x2, line_y2: Line segment end
            line_radius: Thickness of line
        
        Returns:
            Distance to intersection (or -1 if no intersection in ray direction)
        """
        # Vector along line
        line_dx = line_x2 - line_x1
        line_dy = line_y2 - line_y1
        
        # Vector from line start to ray start
        to_ray_x = ray_x - line_x1
        to_ray_y = ray_y - line_y1
        
        # Length squared of line
        line_len_sq = line_dx*line_dx + line_dy*line_dy
        
        if line_len_sq < 1e-6:
            # Line is a point
            return self._ray_point_distance(ray_x, ray_y, ray_dx, ray_dy, 
                                           line_x1, line_y1, line_radius)
        
        # Find closest point on line segment to ray
        t = max(0.0, min(1.0, (to_ray_x * line_dx + to_ray_y * line_dy) / line_len_sq))
        
        closest_x = line_x1 + t * line_dx
        closest_y = line_y1 + t * line_dy
        
        # Distance from ray start to closest point
        to_point_x = closest_x - ray_x
        to_point_y = closest_y - ray_y
        
        # Project onto ray direction
        proj_dist = to_point_x * ray_dx + to_point_y * ray_dy
        
        if proj_dist < 0:
            # Point is behind ray
            return -1
        
        # Perpendicular distance to ray
        perp_dist = abs(to_point_x * ray_dy - to_point_y * ray_dx)
        
        if perp_dist > line_radius:
            # No intersection
            return -1
        
        # Return distance to intersection
        return proj_dist
    
    def _ray_point_distance(self, ray_x: float, ray_y: float, ray_dx: float, ray_dy: float,
                           point_x: float, point_y: float, radius: float) -> float:
        """Calculate distance from ray start to closest point on sphere around point."""
        to_point_x = point_x - ray_x
        to_point_y = point_y - ray_y
        
        proj_dist = to_point_x * ray_dx + to_point_y * ray_dy
        
        if proj_dist < -radius:
            # Point is behind ray
            return -1
        
        perp_dist = abs(to_point_x * ray_dy - to_point_y * ray_dx)
        
        if perp_dist > radius:
            return -1
        
        # Distance to intersection (accounting for sphere radius)
        return max(0.0, proj_dist - math.sqrt(radius*radius - perp_dist*perp_dist))
    
    def get_readings(self, robot_x: float, robot_y: float, robot_angle_rad: float, 
                    obstacles: list) -> list:
        """
        Get current sensor readings.
        
        Args:
            robot_x: Robot X position
            robot_y: Robot Y position
            robot_angle_rad: Robot orientation in radians
            obstacles: List of obstacles
        
        Returns:
            List of normalized sensor readings [0, 1]
        """
        readings = []
        
        for sensor_angle_rel in self.sensor_angles_rad:
            # Absolute angle (robot angle + sensor relative angle)
            absolute_angle = robot_angle_rad + sensor_angle_rel
            
            # Cast ray and get reading
            reading = self.raycast_to_obstacle(robot_x, robot_y, absolute_angle, obstacles)
            readings.append(reading)
        
        self.readings = readings
        return readings
    
    def get_num_sensors(self) -> int:
        """Return number of sensors."""
        return len(self.sensor_angles)
