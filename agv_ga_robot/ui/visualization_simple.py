"""
Simple real-time visualization for AGV GA training.

Single window showing overlayed trajectories:
- Leader (prev gen best) - green
- Top 3 from current gen - blue, cyan, magenta
"""

import os
import sys
import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_ga_robot.maps.map_loader import MapLoader


class SimpleVisualization:
    """Single-window visualization with overlayed robot trajectories."""
    
    # Colors
    COLOR_BG = (20, 20, 20)
    COLOR_GRID = (50, 50, 50)
    COLOR_OBSTACLE = (100, 100, 100)
    COLOR_LEADER = (0, 255, 0)      # Green - leader
    COLOR_RANK1 = (0, 150, 255)     # Blue - rank 1
    COLOR_RANK2 = (0, 255, 255)     # Cyan - rank 2
    COLOR_RANK3 = (255, 0, 255)     # Magenta - rank 3
    COLOR_CHECKPOINT_UNVISITED = (255, 200, 0)  # Orange
    COLOR_CHECKPOINT_VISITED = (0, 255, 0)      # Green
    COLOR_TEXT = (200, 200, 200)
    COLOR_HEADER = (100, 200, 255)
    
    def __init__(self, 
                 map_data: Dict,
                 config: Dict,
                 window_width: int = 1200,
                 window_height: int = 900):
        """
        Initialize simple visualization.
        
        Args:
            map_data: Map configuration
            config: Global configuration
            window_width: Window size
            window_height: Window size
        """
        pygame.init()
        
        self.map_data = map_data
        self.config = config
        self.window_width = window_width
        self.window_height = window_height
        
        self.display = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("AGV GA Training - Live Visualization")
        
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 14)
        self.font_medium = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 24)
        
        # Map dimensions
        self.map_width = map_data['metadata'].get('canvas_width') or map_data['metadata'].get('width', 1200)
        self.map_height = map_data['metadata'].get('canvas_height') or map_data['metadata'].get('height', 800)
        
        # Padding
        self.padding = 40
        self.canvas_width = window_width - 2 * self.padding
        self.canvas_height = window_height - 2 * self.padding - 60  # Leave space for info
        
        # Scaling
        self.scale_x = self.canvas_width / self.map_width
        self.scale_y = self.canvas_height / self.map_height
        
        # State
        self.generation = 0
        self.best_fitness = 0.0
        self.population_avg_fitness = 0.0
        
        # Current trajectories
        self.leader_trajectory = []
        self.rank1_trajectory = []
        self.rank2_trajectory = []
        self.rank3_trajectory = []
        
        # Current stats
        self.leader_stats = {}
        self.top3_stats = [{}, {}, {}]
        
    def map_to_screen(self, map_x: float, map_y: float) -> Tuple[int, int]:
        """Convert map coordinates to screen coordinates."""
        screen_x = int(self.padding + map_x * self.scale_x)
        screen_y = int(self.padding + map_y * self.scale_y)
        return screen_x, screen_y
    
    def draw_map(self) -> None:
        """Draw map elements (obstacles, checkpoints)."""
        # Canvas background
        pygame.draw.rect(self.display, self.COLOR_BG,
                        (self.padding, self.padding,
                         self.canvas_width, self.canvas_height))
        pygame.draw.rect(self.display, self.COLOR_GRID,
                        (self.padding, self.padding,
                         self.canvas_width, self.canvas_height), 2)
        
        # Draw obstacles
        for obstacle in self.map_data['obstacles']:
            points = obstacle['points']
            for i in range(len(points) - 1):
                p1 = self.map_to_screen(points[i][0], points[i][1])
                p2 = self.map_to_screen(points[i+1][0], points[i+1][1])
                pygame.draw.line(self.display, self.COLOR_OBSTACLE, p1, p2, 
                               max(2, int(obstacle.get('width', 5) * self.scale_x)))
        
        # Draw checkpoints
        for cp in self.map_data['checkpoints']:
            pos = self.map_to_screen(cp['x'], cp['y'])
            radius = max(3, int(cp['radius'] * self.scale_x))
            
            # Check if visited (simple heuristic)
            is_visited = False
            color = self.COLOR_CHECKPOINT_VISITED if is_visited else self.COLOR_CHECKPOINT_UNVISITED
            pygame.draw.circle(self.display, color, pos, radius, 2)
    
    def draw_trajectories(self) -> None:
        """Draw all overlayed trajectories."""
        colors_and_trajs = [
            (self.COLOR_LEADER, self.leader_trajectory),
            (self.COLOR_RANK1, self.rank1_trajectory),
            (self.COLOR_RANK2, self.rank2_trajectory),
            (self.COLOR_RANK3, self.rank3_trajectory),
        ]
        
        for color, trajectory in colors_and_trajs:
            if len(trajectory) > 1:
                for i in range(len(trajectory) - 1):
                    p1 = self.map_to_screen(trajectory[i][0], trajectory[i][1])
                    p2 = self.map_to_screen(trajectory[i+1][0], trajectory[i+1][1])
                    pygame.draw.line(self.display, color, p1, p2, 2)
        
        # Draw current positions as circles
        positions = [
            (self.COLOR_LEADER, self.leader_trajectory),
            (self.COLOR_RANK1, self.rank1_trajectory),
            (self.COLOR_RANK2, self.rank2_trajectory),
            (self.COLOR_RANK3, self.rank3_trajectory),
        ]
        
        for color, trajectory in positions:
            if trajectory:
                pos = self.map_to_screen(trajectory[-1][0], trajectory[-1][1])
                pygame.draw.circle(self.display, color, pos, 6)
    
    def draw_info(self) -> None:
        """Draw information panel."""
        info_y = self.padding + self.canvas_height + 10
        
        # Header
        header = f"Gen {self.generation} | Best: {self.best_fitness:.0f} | Avg: {self.population_avg_fitness:.0f}"
        header_surf = self.font_large.render(header, True, self.COLOR_HEADER)
        self.display.blit(header_surf, (self.padding, info_y))
        
        # Legend
        legend_items = [
            (self.COLOR_LEADER, "Leader"),
            (self.COLOR_RANK1, "Rank 1"),
            (self.COLOR_RANK2, "Rank 2"),
            (self.COLOR_RANK3, "Rank 3"),
        ]
        
        legend_y = info_y + 25
        for i, (color, label) in enumerate(legend_items):
            x = self.padding + i * 200
            pygame.draw.line(self.display, color, (x, legend_y), (x + 20, legend_y), 3)
            label_surf = self.font_small.render(label, True, self.COLOR_TEXT)
            self.display.blit(label_surf, (x + 25, legend_y - 6))
    
    def update_episode(self,
                      leader_trajectory: List[Tuple[float, float]],
                      leader_stats: Dict,
                      top3_trajectories: List[List[Tuple[float, float]]],
                      top3_stats: List[Dict]) -> None:
        """
        Update visualization with new episode data.
        
        Args:
            leader_trajectory: Robot path for leader
            leader_stats: Episode stats for leader
            top3_trajectories: List of 3 trajectories
            top3_stats: List of 3 stats dicts
        """
        self.leader_trajectory = leader_trajectory
        self.leader_stats = leader_stats
        
        self.rank1_trajectory = top3_trajectories[0] if len(top3_trajectories) > 0 else []
        self.rank2_trajectory = top3_trajectories[1] if len(top3_trajectories) > 1 else []
        self.rank3_trajectory = top3_trajectories[2] if len(top3_trajectories) > 2 else []
        
        self.top3_stats = top3_stats
        
        # Render frame
        self.display.fill(self.COLOR_BG)
        self.draw_map()
        self.draw_trajectories()
        self.draw_info()
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def update_generation(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
        """Update generation info."""
        self.generation = generation
        self.best_fitness = best_fitness
        self.population_avg_fitness = avg_fitness
    
    def handle_events(self) -> bool:
        """
        Handle pygame events.
        
        Returns:
            True to continue, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
        return True
    
    def close(self) -> None:
        """Clean up pygame."""
        pygame.quit()


# Alias for compatibility
Visualization = SimpleVisualization
