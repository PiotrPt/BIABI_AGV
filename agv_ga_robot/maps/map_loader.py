"""Map Loader - Loads maps from JSON files."""

import json
from typing import Dict, List, Tuple, Any


class MapLoader:
    """Load and validate maps in JSON format."""
    
    @staticmethod
    def load_map(filepath: str) -> Dict[str, Any]:
        """
        Load map from JSON file.
        
        Args:
            filepath: Path to JSON map file
        
        Returns:
            Dictionary with map data
        
        Raises:
            FileNotFoundError: If file not found
            json.JSONDecodeError: If JSON is invalid
            ValueError: If map structure is invalid
        """
        with open(filepath, 'r') as f:
            map_data = json.load(f)
        
        # Validate map structure
        MapLoader._validate_map(map_data)
        return map_data
    
    @staticmethod
    def save_map(filepath: str, map_data: Dict[str, Any]) -> None:
        """
        Save map to JSON file.
        
        Args:
            filepath: Path to save JSON map
            map_data: Dictionary with map data
        """
        MapLoader._validate_map(map_data)
        
        with open(filepath, 'w') as f:
            json.dump(map_data, f, indent=2)
    
    @staticmethod
    def _validate_map(map_data: Dict) -> None:
        """
        Validate map structure.
        
        Args:
            map_data: Map data dictionary
        
        Raises:
            ValueError: If structure is invalid
        """
        required_keys = ['metadata', 'robot', 'checkpoints', 'obstacles']
        
        for key in required_keys:
            if key not in map_data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate metadata
        meta = map_data['metadata']
        if not isinstance(meta, dict):
            raise ValueError("metadata must be a dictionary")
        if 'name' not in meta or 'width' not in meta or 'height' not in meta:
            raise ValueError("metadata must have 'name', 'width', 'height'")
        
        # Validate robot
        robot = map_data['robot']
        if not isinstance(robot, dict):
            raise ValueError("robot must be a dictionary")
        required_robot = ['start_x', 'start_y', 'start_angle']
        for key in required_robot:
            if key not in robot:
                raise ValueError(f"robot must have '{key}'")
        
        # Validate checkpoints
        checkpoints = map_data['checkpoints']
        if not isinstance(checkpoints, list):
            raise ValueError("checkpoints must be a list")
        if len(checkpoints) == 0:
            raise ValueError("must have at least 1 checkpoint")
        
        for i, cp in enumerate(checkpoints):
            if not isinstance(cp, dict):
                raise ValueError(f"checkpoint {i} must be a dictionary")
            required_cp = ['x', 'y', 'radius']
            for key in required_cp:
                if key not in cp:
                    raise ValueError(f"checkpoint {i} must have '{key}'")
        
        # Validate obstacles
        obstacles = map_data['obstacles']
        if not isinstance(obstacles, list):
            raise ValueError("obstacles must be a list")
        
        for i, obs in enumerate(obstacles):
            if not isinstance(obs, dict):
                raise ValueError(f"obstacle {i} must be a dictionary")
            if 'points' not in obs or 'width' not in obs:
                raise ValueError(f"obstacle {i} must have 'points' and 'width'")
            if not isinstance(obs['points'], list) or len(obs['points']) < 2:
                raise ValueError(f"obstacle {i} must have at least 2 points")
    
    @staticmethod
    def get_robot_start(map_data: Dict) -> Tuple[float, float, float]:
        """
        Get robot starting position and angle.
        
        Args:
            map_data: Map data
        
        Returns:
            (start_x, start_y, start_angle)
        """
        robot = map_data['robot']
        return (robot['start_x'], robot['start_y'], robot['start_angle'])
    
    @staticmethod
    def get_checkpoints(map_data: Dict) -> List[Dict]:
        """
        Get checkpoints list.
        
        Args:
            map_data: Map data
        
        Returns:
            List of checkpoint dictionaries
        """
        return map_data['checkpoints']
    
    @staticmethod
    def get_obstacles(map_data: Dict) -> List[Dict]:
        """
        Get obstacles list.
        
        Args:
            map_data: Map data
        
        Returns:
            List of obstacle dictionaries
        """
        return map_data['obstacles']
    
    @staticmethod
    def get_resolution(map_data: Dict) -> Tuple[int, int]:
        """
        Get map resolution.
        
        Args:
            map_data: Map data
        
        Returns:
            (width, height)
        """
        meta = map_data['metadata']
        return (meta['width'], meta['height'])
