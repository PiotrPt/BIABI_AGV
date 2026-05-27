"""
Interactive map editor for AGV environment.

Features:
- Toolbar buttons: Draw/Checkpoint/Start/Undo/Clear/Preview/Save/Load
- Draw obstacles with mouse (LMB drag = line segment)
- Place checkpoints and start position
- Save/load maps to JSON format
- Preview mode to test current map
"""

import os
import sys
import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_ga_robot.maps.map_loader import MapLoader
from agv_ga_robot.env.agv_env import AGVEnvironment
from agv_ga_robot.utils.helpers import load_config


class MapEditor:
    """Interactive map editor with Pygame GUI."""
    
    # Colors
    COLOR_BG = (40, 40, 40)
    COLOR_GRID = (60, 60, 60)
    COLOR_OBSTACLE = (100, 100, 100)
    COLOR_CHECKPOINT = (0, 150, 255)
    COLOR_ROBOT_START = (0, 255, 0)
    COLOR_BUTTON_IDLE = (80, 80, 80)
    COLOR_BUTTON_HOVER = (120, 120, 120)
    COLOR_BUTTON_ACTIVE = (200, 100, 100)
    COLOR_TEXT = (200, 200, 200)
    
    # Tools
    TOOL_DRAW = "draw"
    TOOL_CHECKPOINT = "checkpoint"
    TOOL_START = "start"
    TOOL_PREVIEW = "preview"
    
    def __init__(self, 
                 config_path: str,
                 window_width: int = 1400,
                 window_height: int = 900,
                 canvas_width: int = 1200,
                 canvas_height: int = 800):
        """
        Initialize map editor.
        
        Args:
            config_path: Path to config.yaml (for default parameters)
            window_width: Window size
            window_height: Window size
            canvas_width: Map canvas size
            canvas_height: Map canvas size
        """
        pygame.init()
        
        self.config_path = config_path
        self.config = load_config(config_path)
        
        self.window_width = window_width
        self.window_height = window_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # GUI layout
        self.toolbar_height = 50
        self.sidebar_width = 150
        self.canvas_x = self.sidebar_width
        self.canvas_y = self.toolbar_height
        self.canvas_display_width = self.window_width - self.sidebar_width
        self.canvas_display_height = self.window_height - self.toolbar_height
        
        # Scaling
        self.scale_x = self.canvas_display_width / self.canvas_width
        self.scale_y = self.canvas_display_height / self.canvas_height
        
        self.display = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("AGV Map Editor")
        
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 12)
        self.font_medium = pygame.font.Font(None, 16)
        
        # State
        self.current_tool = self.TOOL_DRAW
        self.current_drawing = []  # Current line being drawn
        self.obstacles = []  # List of obstacle line segments
        self.checkpoints = []  # List of checkpoint positions
        self.robot_start = None  # (x, y, angle)
        
        self.undo_stack = []  # For undo functionality
        self.history = []  # Full history
        
        # Preview mode
        self.preview_mode = False
        self.preview_env = None
        
        # Define buttons
        self.buttons = self._create_buttons()
        
    def _create_buttons(self) -> List[Dict]:
        """Create toolbar buttons."""
        buttons = [
            {'label': 'Draw', 'tool': self.TOOL_DRAW, 'x': 10, 'y': 5, 'w': 40, 'h': 40},
            {'label': 'CP', 'tool': self.TOOL_CHECKPOINT, 'x': 55, 'y': 5, 'w': 40, 'h': 40},
            {'label': 'Start', 'tool': self.TOOL_START, 'x': 100, 'y': 5, 'w': 40, 'h': 40},
            {'label': 'Undo', 'action': 'undo', 'x': 145, 'y': 5, 'w': 40, 'h': 40},
            {'label': 'Clear', 'action': 'clear', 'x': 190, 'y': 5, 'w': 40, 'h': 40},
            {'label': 'Preview', 'action': 'preview', 'x': 235, 'y': 5, 'w': 50, 'h': 40},
            {'label': 'Save', 'action': 'save', 'x': 290, 'y': 5, 'w': 40, 'h': 40},
            {'label': 'Load', 'action': 'load', 'x': 335, 'y': 5, 'w': 40, 'h': 40},
        ]
        return buttons
    
    def canvas_to_map(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to map coordinates."""
        map_x = (screen_x - self.canvas_x) / self.scale_x
        map_y = (screen_y - self.canvas_y) / self.scale_y
        return map_x, map_y
    
    def map_to_canvas(self, map_x: float, map_y: float) -> Tuple[int, int]:
        """Convert map coordinates to screen coordinates."""
        screen_x = int(self.canvas_x + map_x * self.scale_x)
        screen_y = int(self.canvas_y + map_y * self.scale_y)
        return screen_x, screen_y
    
    def draw_canvas(self) -> None:
        """Draw the map canvas."""
        # Background
        pygame.draw.rect(self.display, self.COLOR_BG,
                        (self.canvas_x, self.canvas_y, 
                         self.canvas_display_width, self.canvas_display_height))
        pygame.draw.rect(self.display, self.COLOR_GRID,
                        (self.canvas_x, self.canvas_y,
                         self.canvas_display_width, self.canvas_display_height), 2)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            if len(obstacle) > 1:
                for i in range(len(obstacle) - 1):
                    p1 = self.map_to_canvas(*obstacle[i])
                    p2 = self.map_to_canvas(*obstacle[i+1])
                    pygame.draw.line(self.display, self.COLOR_OBSTACLE, p1, p2, 5)
        
        # Draw current drawing line
        if len(self.current_drawing) > 1:
            for i in range(len(self.current_drawing) - 1):
                p1 = self.map_to_canvas(*self.current_drawing[i])
                p2 = self.map_to_canvas(*self.current_drawing[i+1])
                pygame.draw.line(self.display, (255, 200, 100), p1, p2, 3)
        
        # Draw checkpoints
        for i, cp in enumerate(self.checkpoints):
            pos = self.map_to_canvas(*cp)
            pygame.draw.circle(self.display, self.COLOR_CHECKPOINT, pos, 8, 2)
            # Draw checkpoint number
            num_text = self.font_small.render(str(i+1), True, self.COLOR_CHECKPOINT)
            self.display.blit(num_text, (pos[0] - 3, pos[1] - 3))
        
        # Draw robot start
        if self.robot_start:
            pos = self.map_to_canvas(*self.robot_start[:2])
            pygame.draw.circle(self.display, self.COLOR_ROBOT_START, pos, 8, 2)
            # Draw direction indicator
            angle = self.robot_start[2]
            end_x = pos[0] + 10 * np.cos(angle)
            end_y = pos[1] + 10 * np.sin(angle)
            pygame.draw.line(self.display, self.COLOR_ROBOT_START, pos, (end_x, end_y), 2)
    
    def draw_toolbar(self) -> None:
        """Draw toolbar with buttons."""
        pygame.draw.rect(self.display, self.COLOR_BG, (0, 0, self.window_width, self.toolbar_height))
        pygame.draw.rect(self.display, self.COLOR_GRID, (0, 0, self.window_width, self.toolbar_height), 1)
        
        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            is_hover = (btn['x'] <= mouse_pos[0] <= btn['x'] + btn['w'] and
                       btn['y'] <= mouse_pos[1] <= btn['y'] + btn['h'])
            
            is_active = (btn.get('tool') == self.current_tool)
            
            if is_active:
                color = self.COLOR_BUTTON_ACTIVE
            elif is_hover:
                color = self.COLOR_BUTTON_HOVER
            else:
                color = self.COLOR_BUTTON_IDLE
            
            pygame.draw.rect(self.display, color, (btn['x'], btn['y'], btn['w'], btn['h']), 0)
            pygame.draw.rect(self.display, self.COLOR_GRID, (btn['x'], btn['y'], btn['w'], btn['h']), 1)
            
            # Draw label
            label = self.font_small.render(btn['label'], True, self.COLOR_TEXT)
            label_rect = label.get_rect(center=(btn['x'] + btn['w']//2, btn['y'] + btn['h']//2))
            self.display.blit(label, label_rect)
        
        # Draw status bar
        status_text = f"Tool: {self.current_tool} | Checkpoints: {len(self.checkpoints)} | Obstacles: {len(self.obstacles)}"
        status_surf = self.font_small.render(status_text, True, self.COLOR_TEXT)
        self.display.blit(status_surf, (self.window_width - 300, 15))
    
    def handle_events(self) -> bool:
        """Handle user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.undo()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_left_click(event.pos)
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._handle_left_release()
            
            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # Left button held
                    self._handle_mouse_motion(event.pos)
        
        return True
    
    def _handle_left_click(self, pos: Tuple[int, int]) -> None:
        """Handle left mouse click."""
        # Check toolbar buttons
        for btn in self.buttons:
            if btn['x'] <= pos[0] <= btn['x'] + btn['w'] and \
               btn['y'] <= pos[1] <= btn['y'] + btn['h']:
                # Button clicked
                if 'tool' in btn:
                    self.current_tool = btn['tool']
                elif 'action' in btn:
                    self._handle_action(btn['action'])
                return
        
        # Canvas click
        if pos[0] >= self.canvas_x and pos[1] >= self.canvas_y:
            map_x, map_y = self.canvas_to_map(*pos)
            map_x = np.clip(map_x, 0, self.canvas_width)
            map_y = np.clip(map_y, 0, self.canvas_height)
            
            if self.current_tool == self.TOOL_DRAW:
                self.current_drawing = [(map_x, map_y)]
            elif self.current_tool == self.TOOL_CHECKPOINT:
                self.checkpoints.append((map_x, map_y))
            elif self.current_tool == self.TOOL_START:
                self.robot_start = (map_x, map_y, 0)  # 0 radians
    
    def _handle_left_release(self) -> None:
        """Handle left mouse button release."""
        if self.current_tool == self.TOOL_DRAW and len(self.current_drawing) > 1:
            self.obstacles.append(self.current_drawing.copy())
            self.current_drawing = []
    
    def _handle_mouse_motion(self, pos: Tuple[int, int]) -> None:
        """Handle mouse motion while dragging."""
        if self.current_tool == self.TOOL_DRAW:
            if pos[0] >= self.canvas_x and pos[1] >= self.canvas_y:
                map_x, map_y = self.canvas_to_map(*pos)
                map_x = np.clip(map_x, 0, self.canvas_width)
                map_y = np.clip(map_y, 0, self.canvas_height)
                
                if len(self.current_drawing) == 0 or \
                   (abs(map_x - self.current_drawing[-1][0]) > 1 or
                    abs(map_y - self.current_drawing[-1][1]) > 1):
                    self.current_drawing.append((map_x, map_y))
    
    def _handle_action(self, action: str) -> None:
        """Handle toolbar actions."""
        if action == "undo":
            self.undo()
        elif action == "clear":
            self.obstacles = []
            self.checkpoints = []
            self.robot_start = None
            self.current_drawing = []
        elif action == "preview":
            self.preview_mode = not self.preview_mode
        elif action == "save":
            self.save_map()
        elif action == "load":
            self.load_map()
    
    def undo(self) -> None:
        """Undo last action."""
        if len(self.obstacles) > 0:
            self.obstacles.pop()
    
    def save_map(self, filepath: str = "custom_map.json") -> None:
        """Save current map to JSON."""
        map_data = {
            'metadata': {
                'name': 'Custom Map',
                'canvas_width': self.canvas_width,
                'canvas_height': self.canvas_height,
            },
            'robot': {
                'start_x': self.robot_start[0] if self.robot_start else 100,
                'start_y': self.robot_start[1] if self.robot_start else 100,
                'start_angle_deg': 0,
            },
            'checkpoints': [
                {'id': i, 'x': cp[0], 'y': cp[1], 'radius': 50}
                for i, cp in enumerate(self.checkpoints)
            ],
            'obstacles': [
                {'points': obs, 'width': 15}
                for obs in self.obstacles
            ]
        }
        
        MapLoader.save_map(filepath, map_data)
        print(f"[Saved] Map saved to {filepath}")
    
    def load_map(self, filepath: str = "custom_map.json") -> None:
        """Load map from JSON."""
        try:
            map_data = MapLoader.load_map(filepath)
            
            self.canvas_width = map_data['metadata'].get('canvas_width', 1200)
            self.canvas_height = map_data['metadata'].get('canvas_height', 800)
            
            robot = map_data['robot']
            self.robot_start = (robot['start_x'], robot['start_y'], 0)
            
            self.checkpoints = [(cp['x'], cp['y']) for cp in map_data['checkpoints']]
            self.obstacles = [obs['points'] for obs in map_data['obstacles']]
            
            print(f"[Loaded] Map loaded from {filepath}")
        except Exception as e:
            print(f"[Error] Failed to load map: {e}")
    
    def run(self) -> None:
        """Main event loop."""
        running = True
        
        print("\n" + "="*60)
        print("AGV Map Editor")
        print("="*60)
        print("Controls:")
        print("  Draw   - Draw obstacles with mouse")
        print("  CP     - Add checkpoints")
        print("  Start  - Set robot start position")
        print("  Undo   - Undo last action")
        print("  Clear  - Clear all")
        print("  Preview - Preview map (not yet implemented)")
        print("  Save   - Save map to JSON")
        print("  Load   - Load map from JSON")
        print("="*60 + "\n")
        
        while running:
            running = self.handle_events()
            
            # Clear screen
            self.display.fill(self.COLOR_BG)
            
            # Draw everything
            self.draw_canvas()
            self.draw_toolbar()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Run map editor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGV Map Editor")
    parser.add_argument("--config", type=str,
                       default="agv_ga_robot/config/config.yaml",
                       help="Path to config.yaml")
    parser.add_argument("--output", type=str,
                       default="custom_map.json",
                       help="Output map filename")
    
    args = parser.parse_args()
    
    editor = MapEditor(args.config)
    editor.run()
    editor.save_map(args.output)


if __name__ == "__main__":
    main()
