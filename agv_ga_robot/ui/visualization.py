"""
Real-time visualization for AGV GA training.

Single window with overlayed trajectories:
- Leader (prev gen best) - green
- Top 3 from current gen - blue/cyan/magenta
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_ga_robot.maps.map_loader import MapLoader




class Visualization:
    """Single-window visualization with overlayed robot trajectories using matplotlib."""
    
    def __init__(self, 
                 map_data: Dict,
                 config: Dict,
                 window_width: int = 14,
                 window_height: int = 10):
        """
        Initialize visualization.
        
        Args:
            map_data: Map configuration
            config: Global configuration
            window_width: Figure width in inches
            window_height: Figure height in inches
        """
        self.map_data = map_data
        self.config = config
        
        # Map dimensions
        self.map_width = map_data['metadata'].get('canvas_width') or map_data['metadata'].get('width', 1200)
        self.map_height = map_data['metadata'].get('canvas_height') or map_data['metadata'].get('height', 800)
        
        # Create figure with interactive mode
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(window_width, window_height))
        self.fig.suptitle("AGV GA Training - Live Visualization", fontsize=16)
        
        # State
        self.generation = 0
        self.best_fitness = 0.0
        self.population_avg_fitness = 0.0
        
        # Current trajectories (ensure always 4)
        self.leader_trajectory = []
        self.rank1_trajectory = []
        self.rank2_trajectory = []
        self.rank3_trajectory = []
        
        # Setup axes
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(self.map_height, 0)  # Invert Y axis
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (pixels)", fontsize=10)
        self.ax.set_ylabel("Y (pixels)", fontsize=10)
        
        self.fig.tight_layout()
    
    def draw_map(self) -> None:
        """Draw map elements (obstacles, checkpoints)."""
        self.ax.clear()
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(self.map_height, 0)  # Invert Y axis
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (pixels)", fontsize=10)
        self.ax.set_ylabel("Y (pixels)", fontsize=10)
        
        # Draw obstacles
        for obstacle in self.map_data.get('obstacles', []):
            points = obstacle.get('points', [])
            if len(points) > 1:
                for i in range(len(points) - 1):
                    x_vals = [points[i][0], points[i+1][0]]
                    y_vals = [points[i][1], points[i+1][1]]
                    width = obstacle.get('width', 5)
                    self.ax.plot(x_vals, y_vals, 'gray', linewidth=width, solid_capstyle='round')
        
        # Draw checkpoints
        for cp in self.map_data.get('checkpoints', []):
            circle = patches.Circle((cp['x'], cp['y']), cp['radius'], 
                                   fill=False, edgecolor='orange', linewidth=2)
            self.ax.add_patch(circle)
    
    def draw_trajectories(self) -> None:
        """Draw all overlayed trajectories."""
        trajectories = [
            ('red', self.leader_trajectory, 'Leader', 6),      # Czerwony, grubsza linia (6px)
            ('blue', self.rank1_trajectory, 'Rank 1', 2),
            ('cyan', self.rank2_trajectory, 'Rank 2', 2),
            ('magenta', self.rank3_trajectory, 'Rank 3', 2),
        ]
        
        for color, trajectory, label, linewidth in trajectories:
            if trajectory and len(trajectory) > 1:
                try:
                    x_vals = [float(t[0]) for t in trajectory]
                    y_vals = [float(t[1]) for t in trajectory]
                    self.ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, label=label, alpha=0.8)
                    
                    # Mark current position
                    self.ax.plot(x_vals[-1], y_vals[-1], 'o', color=color, markersize=12)
                except Exception as e:
                    print(f"  [Warning] Error plotting {label}: {e}")
    
    def draw_info(self) -> None:
        """Draw information panel."""
        info_text = f"Gen {self.generation} | Best: {self.best_fitness:.0f} | Avg: {self.population_avg_fitness:.0f}"
        self.ax.text(0.5, -0.10, info_text, transform=self.ax.transAxes,
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def update_episode(self,
                      leader_trajectory: List[Tuple[float, float]],
                      leader_stats: Dict,
                      top3_trajectories: List[List[Tuple[float, float]]],
                      top3_stats: List[Dict]) -> None:
        """
        Update visualization with new episode data.
        """
        # Safely assign trajectories
        self.leader_trajectory = leader_trajectory if leader_trajectory else []
        
        self.rank1_trajectory = top3_trajectories[0] if (top3_trajectories and len(top3_trajectories) > 0) else []
        self.rank2_trajectory = top3_trajectories[1] if (top3_trajectories and len(top3_trajectories) > 1) else []
        self.rank3_trajectory = top3_trajectories[2] if (top3_trajectories and len(top3_trajectories) > 2) else []
        
        try:
            self.draw_map()
            self.draw_trajectories()
            self.draw_info()
            
            self.ax.legend(loc='upper right', fontsize=9)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"  [Warning] Visualization update error: {e}")
    
    def update_generation(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
        """Update generation info."""
        self.generation = generation
        self.best_fitness = best_fitness
        self.population_avg_fitness = avg_fitness
    
    def handle_events(self) -> bool:
        """Handle window events. Returns True to continue."""
        try:
            self.fig.canvas.flush_events()
            if not plt.fignum_exists(self.fig.number):
                return False
            return True
        except:
            return True  # Continue even if there's an error
    
    def close(self) -> None:
        """Close visualization."""
        try:
            plt.close(self.fig)
        except:
            pass

