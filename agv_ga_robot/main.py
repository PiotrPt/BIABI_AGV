"""
Main launcher GUI for AGV GA Robot.

Provides user interface to select:
- Training mode: choose map, set population/generations, start training
- Replay mode: choose map, select genome file, watch animation
"""

import os
import sys
import pygame
from pathlib import Path
from typing import Optional, Tuple, Dict
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agv_ga_robot.utils.helpers import load_config
from agv_ga_robot.maps.map_loader import MapLoader


class MainGUI:
    """Main launcher GUI using pygame."""
    
    # Colors
    COLOR_BG = (30, 30, 30)
    COLOR_PANEL = (50, 50, 50)
    COLOR_BUTTON_IDLE = (70, 130, 180)
    COLOR_BUTTON_HOVER = (100, 160, 220)
    COLOR_BUTTON_ACTIVE = (150, 200, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEADER = (100, 200, 255)
    COLOR_INPUT_BG = (60, 60, 60)
    COLOR_INPUT_BORDER = (100, 100, 100)
    COLOR_INPUT_ACTIVE = (150, 180, 220)
    FONT_CANDIDATES = ["segoeui", "arial", "dejavusans"]
    
    def __init__(self, config_path: str = "agv_ga_robot/config/config.yaml"):
        """
        Initialize main GUI.
        
        Args:
            config_path: Path to configuration file
        """
        pygame.init()
        
        self.config = load_config(config_path)
        self.window_width = 1120
        self.window_height = 800
        self.display = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("AGV GA Robot - Main Launcher")
        self.project_root = Path(__file__).resolve().parent.parent
        
        self.clock = pygame.time.Clock()
        self.font_path = pygame.font.match_font(self.FONT_CANDIDATES)
        self.font_title = self._make_font(44, bold=True)
        self.font_large = self._make_font(30, bold=True)
        self.font_medium = self._make_font(24, bold=True)
        self.font_small = self._make_font(20)
        self.font_tiny = self._make_font(17)
        self.font_hint = self._make_font(15)
        
        # Available maps
        self.maps = self.config.get('MAPS_AVAILABLE', 
                                    ['learning_maze_simple', 'learning_maze_winding', 'learning_maze_90deg'])
        self.maps_with_extensions = [f"agv_ga_robot/maps/{m}.json" for m in self.maps]
        
        # Default training parameters
        self.default_population = self.config.get('TRAIN_PARAMS', {}).get('default_population_size', 180)
        self.default_generations = self.config.get('TRAIN_PARAMS', {}).get('default_num_generations', 250)
        
        # State
        self.current_mode = None  # 'train' or 'replay'
        self.selected_map_idx = 0
        self.population_input = str(self.default_population)
        self.generations_input = str(self.default_generations)
        self.selected_genome_path = None
        self.active_input = None  # 'population' or 'generations' or None
        
        # Buttons
        self.train_button = pygame.Rect(50, 110, 490, 72)
        self.replay_button = pygame.Rect(580, 110, 490, 72)
        
        # Train mode elements
        self.map_dropdown_rect = pygame.Rect(50, 230, 490, 46)
        self.population_input_rect = pygame.Rect(50, 300, 490, 46)
        self.generations_input_rect = pygame.Rect(50, 370, 490, 46)
        self.start_train_button = pygame.Rect(50, 450, 490, 58)
        
        # Replay mode elements
        self.replay_map_dropdown_rect = pygame.Rect(580, 230, 490, 46)
        self.genome_selector_rect = pygame.Rect(580, 300, 490, 46)
        self.genome_file_list = self._get_genome_files()
        self.genome_list_scroll = 0
        self.genome_list_visible = []
        self.genome_list_rect = pygame.Rect(580, 370, 490, 180)
        self.start_replay_button = pygame.Rect(580, 570, 490, 58)
        
        # Back button (visible in mode selection)
        self.back_button = pygame.Rect(50, 670, 1020, 56)

    def _make_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """Create a readable UI font with a sane system fallback."""
        if self.font_path:
            return pygame.font.Font(self.font_path, size)
        return pygame.font.Font(None, size)
        
    def _get_genome_files(self) -> list:
        """Get list of available final genome files for replay."""
        genomes = []

        final_dir = self.project_root / 'training_results_final'
        if final_dir.exists():
            for fname in final_dir.iterdir():
                if fname.suffix == '.pkl' and fname.name.startswith('best_genome_final_'):
                    genomes.append(('training_results_final', fname.name))

        def sort_key(item):
            fname = item[1]
            stem = Path(fname).stem
            try:
                return int(stem.rsplit('_', 1)[-1])
            except ValueError:
                return -1

        return sorted(genomes, key=sort_key, reverse=True)
    
    def _draw_mode_selection(self):
        """Draw initial mode selection screen."""
        self.display.fill(self.COLOR_BG)
        
        # Title
        title = self.font_title.render("AGV GA Robot Launcher", True, self.COLOR_HEADER)
        title_rect = title.get_rect(center=(self.window_width // 2, 34))
        self.display.blit(title, title_rect)
        
        # Subtitle
        subtitle = self.font_small.render("Select a mode to continue", True, self.COLOR_TEXT)
        subtitle_rect = subtitle.get_rect(center=(self.window_width // 2, 82))
        self.display.blit(subtitle, subtitle_rect)
        
        # Train button
        train_hover = self.train_button.collidepoint(pygame.mouse.get_pos())
        train_color = self.COLOR_BUTTON_HOVER if train_hover else self.COLOR_BUTTON_IDLE
        pygame.draw.rect(self.display, train_color, self.train_button, border_radius=8)
        pygame.draw.rect(self.display, self.COLOR_BUTTON_ACTIVE, self.train_button, 3, border_radius=8)
        train_text = self.font_medium.render("TRAINING MODE", True, (255, 255, 255))
        train_text_rect = train_text.get_rect(center=self.train_button.center)
        self.display.blit(train_text, train_text_rect)
        train_hint = self.font_hint.render("Start learning with a selected map and custom population/generations", True, self.COLOR_TEXT)
        self.display.blit(train_hint, (self.train_button.x + 18, self.train_button.y + 42))
        
        # Replay button
        replay_hover = self.replay_button.collidepoint(pygame.mouse.get_pos())
        replay_color = self.COLOR_BUTTON_HOVER if replay_hover else self.COLOR_BUTTON_IDLE
        pygame.draw.rect(self.display, replay_color, self.replay_button, border_radius=8)
        pygame.draw.rect(self.display, self.COLOR_BUTTON_ACTIVE, self.replay_button, 3, border_radius=8)
        replay_text = self.font_medium.render("REPLAY MODE", True, (255, 255, 255))
        replay_text_rect = replay_text.get_rect(center=self.replay_button.center)
        self.display.blit(replay_text, replay_text_rect)
        replay_hint = self.font_hint.render("Pick a genome and watch the robot move step by step", True, self.COLOR_TEXT)
        self.display.blit(replay_hint, (self.replay_button.x + 18, self.replay_button.y + 42))
    
    def _draw_train_mode(self):
        """Draw training mode configuration screen."""
        self.display.fill(self.COLOR_BG)
        
        # Title
        title = self.font_title.render("Training Configuration", True, self.COLOR_HEADER)
        title_rect = title.get_rect(center=(self.window_width // 2, 34))
        self.display.blit(title, title_rect)
        
        # Map selection
        map_label = self.font_small.render("Select map:", True, self.COLOR_TEXT)
        self.display.blit(map_label, (50, 196))
        
        # Map dropdown
        pygame.draw.rect(self.display, self.COLOR_INPUT_BG, self.map_dropdown_rect, border_radius=4)
        pygame.draw.rect(self.display, self.COLOR_INPUT_BORDER, self.map_dropdown_rect, 2, border_radius=4)
        map_text = self.font_small.render(self.maps[self.selected_map_idx], True, self.COLOR_TEXT)
        self.display.blit(map_text, (self.map_dropdown_rect.x + 12, self.map_dropdown_rect.y + 11))
        
        # Population input
        pop_label = self.font_small.render("Population size:", True, self.COLOR_TEXT)
        self.display.blit(pop_label, (50, 266))
        
        pop_active = self.active_input == 'population'
        pop_color = self.COLOR_INPUT_ACTIVE if pop_active else self.COLOR_INPUT_BG
        pygame.draw.rect(self.display, pop_color, self.population_input_rect, border_radius=4)
        pygame.draw.rect(self.display, self.COLOR_INPUT_BORDER, self.population_input_rect, 2, border_radius=4)
        pop_text = self.font_small.render(self.population_input, True, self.COLOR_TEXT)
        self.display.blit(pop_text, (self.population_input_rect.x + 12, self.population_input_rect.y + 11))
        
        # Generations input
        gen_label = self.font_small.render("Generations:", True, self.COLOR_TEXT)
        self.display.blit(gen_label, (50, 336))
        
        gen_active = self.active_input == 'generations'
        gen_color = self.COLOR_INPUT_ACTIVE if gen_active else self.COLOR_INPUT_BG
        pygame.draw.rect(self.display, gen_color, self.generations_input_rect, border_radius=4)
        pygame.draw.rect(self.display, self.COLOR_INPUT_BORDER, self.generations_input_rect, 2, border_radius=4)
        gen_text = self.font_small.render(self.generations_input, True, self.COLOR_TEXT)
        self.display.blit(gen_text, (self.generations_input_rect.x + 12, self.generations_input_rect.y + 11))
        
        # Start training button
        pygame.draw.rect(self.display, self.COLOR_BUTTON_ACTIVE, self.start_train_button, border_radius=8)
        pygame.draw.rect(self.display, self.COLOR_HEADER, self.start_train_button, 3, border_radius=8)
        start_text = self.font_medium.render("START TRAINING", True, (255, 255, 255))
        start_text_rect = start_text.get_rect(center=self.start_train_button.center)
        self.display.blit(start_text, start_text_rect)
        
        # Back button
        self._draw_back_button()
    
    def _draw_replay_mode(self):
        """Draw replay mode configuration screen."""
        self.display.fill(self.COLOR_BG)
        
        # Title
        title = self.font_title.render("Replay Configuration", True, self.COLOR_HEADER)
        title_rect = title.get_rect(center=(self.window_width // 2, 34))
        self.display.blit(title, title_rect)
        
        # Map selection
        map_label = self.font_small.render("Select map:", True, self.COLOR_TEXT)
        self.display.blit(map_label, (580, 196))
        
        # Map dropdown
        pygame.draw.rect(self.display, self.COLOR_INPUT_BG, self.replay_map_dropdown_rect, border_radius=4)
        pygame.draw.rect(self.display, self.COLOR_INPUT_BORDER, self.replay_map_dropdown_rect, 2, border_radius=4)
        map_text = self.font_small.render(self.maps[self.selected_map_idx], True, self.COLOR_TEXT)
        self.display.blit(map_text, (self.replay_map_dropdown_rect.x + 12, self.replay_map_dropdown_rect.y + 11))
        
        # Genome selector
        genome_label = self.font_small.render("Select genome:", True, self.COLOR_TEXT)
        self.display.blit(genome_label, (580, 266))
        
        pygame.draw.rect(self.display, self.COLOR_INPUT_BG, self.genome_selector_rect, border_radius=4)
        pygame.draw.rect(self.display, self.COLOR_INPUT_BORDER, self.genome_selector_rect, 2, border_radius=4)
        if self.selected_genome_path:
            genome_display = os.path.basename(self.selected_genome_path)
        else:
            genome_display = "Choose genome..."
        genome_text = self.font_tiny.render(genome_display, True, self.COLOR_TEXT)
        self.display.blit(genome_text, (self.genome_selector_rect.x + 12, self.genome_selector_rect.y + 13))
        
        # Genome list
        pygame.draw.rect(self.display, self.COLOR_INPUT_BG, self.genome_list_rect, border_radius=4)
        pygame.draw.rect(self.display, self.COLOR_INPUT_BORDER, self.genome_list_rect, 2, border_radius=4)
        
        # Render genome list items
        item_height = 30
        visible_items = self.genome_list_rect.height // item_height
        self.genome_list_visible = self.genome_file_list[self.genome_list_scroll:self.genome_list_scroll + visible_items]
        
        for i, (folder, fname) in enumerate(self.genome_list_visible):
            y = self.genome_list_rect.y + i * item_height
            item_rect = pygame.Rect(self.genome_list_rect.x, y, self.genome_list_rect.width, item_height)
            
            # Highlight if selected
            if self.selected_genome_path and fname in self.selected_genome_path:
                pygame.draw.rect(self.display, self.COLOR_BUTTON_HOVER, item_rect)
            
            item_text = self.font_tiny.render(f"{folder}/{fname}", True, self.COLOR_TEXT)
            self.display.blit(item_text, (item_rect.x + 10, item_rect.y + 7))
        
        # Start replay button
        pygame.draw.rect(self.display, self.COLOR_BUTTON_ACTIVE, self.start_replay_button, border_radius=8)
        pygame.draw.rect(self.display, self.COLOR_HEADER, self.start_replay_button, 3, border_radius=8)
        start_text = self.font_medium.render("START REPLAY", True, (255, 255, 255))
        start_text_rect = start_text.get_rect(center=self.start_replay_button.center)
        self.display.blit(start_text, start_text_rect)
        
        # Back button
        self._draw_back_button()
    
    def _draw_back_button(self):
        """Draw back button."""
        back_hover = self.back_button.collidepoint(pygame.mouse.get_pos())
        back_color = self.COLOR_BUTTON_HOVER if back_hover else self.COLOR_BUTTON_IDLE
        pygame.draw.rect(self.display, back_color, self.back_button, border_radius=8)
        pygame.draw.rect(self.display, self.COLOR_BUTTON_ACTIVE, self.back_button, 2, border_radius=8)
        back_text = self.font_medium.render("BACK", True, self.COLOR_TEXT)
        back_text_rect = back_text.get_rect(center=self.back_button.center)
        self.display.blit(back_text, back_text_rect)
    
    def _handle_train_clicks(self, pos: Tuple[int, int]):
        """Handle mouse clicks in training mode."""
        # Map dropdown
        if self.map_dropdown_rect.collidepoint(pos):
            self.selected_map_idx = (self.selected_map_idx + 1) % len(self.maps)
        
        # Population input
        elif self.population_input_rect.collidepoint(pos):
            self.active_input = 'population'
            self.population_input = ''
        
        # Generations input
        elif self.generations_input_rect.collidepoint(pos):
            self.active_input = 'generations'
            self.generations_input = ''
        
        # Start training button
        elif self.start_train_button.collidepoint(pos):
            self._start_training()
        
        # Back button
        elif self.back_button.collidepoint(pos):
            self.current_mode = None
            self.active_input = None
    
    def _handle_replay_clicks(self, pos: Tuple[int, int]):
        """Handle mouse clicks in replay mode."""
        # Map dropdown
        if self.replay_map_dropdown_rect.collidepoint(pos):
            self.selected_map_idx = (self.selected_map_idx + 1) % len(self.maps)
        
        # Genome list clicks
        elif self.genome_list_rect.collidepoint(pos):
            item_height = 25
            clicked_idx = (pos[1] - self.genome_list_rect.y) // item_height
            if clicked_idx < len(self.genome_list_visible):
                folder, fname = self.genome_list_visible[clicked_idx]
                self.selected_genome_path = os.path.join(folder, fname)
        
        # Start replay button
        elif self.start_replay_button.collidepoint(pos):
            self._start_replay()
        
        # Back button
        elif self.back_button.collidepoint(pos):
            self.current_mode = None
    
    def _start_training(self):
        """Start training with current configuration."""
        try:
            map_name = self.maps[self.selected_map_idx]
            population = int(self.population_input)
            generations = int(self.generations_input)
            
            map_path = f"agv_ga_robot/maps/{map_name}.json"
            
            # Validate map exists
            if not os.path.exists(map_path):
                print(f"[Error] Map not found: {map_path}")
                return
            
            print(f"\n[Starting Training]")
            print(f"  Map: {map_name}")
            print(f"  Population: {population}")
            print(f"  Generations: {generations}")
            
            # Run training in subprocess so GUI stays responsive
            cmd = [
                sys.executable, "-m", "agv_ga_robot.main_train",
                "--map", map_name,
                "--population", str(population),
                "--generations", str(generations)
            ]
            
            subprocess.Popen(cmd, cwd=str(self.project_root))
            
            # Optionally close GUI after starting
            # For now, let user close manually or continue to other options
            print("[Info] Training started in separate window")
            
        except ValueError:
            print("[Error] Invalid population or generations value")
    
    def _start_replay(self):
        """Start replay with selected genome."""
        if not self.selected_genome_path:
            print("[Error] No genome selected")
            return
        
        try:
            map_name = self.maps[self.selected_map_idx]
            map_path = f"agv_ga_robot/maps/{map_name}.json"
            
            # Validate
            if not os.path.exists(map_path):
                print(f"[Error] Map not found: {map_path}")
                return
            
            if not os.path.exists(self.selected_genome_path):
                print(f"[Error] Genome not found: {self.selected_genome_path}")
                return
            
            print(f"\n[Starting Replay]")
            print(f"  Map: {map_name}")
            print(f"  Genome: {self.selected_genome_path}")
            
            # Run replay in subprocess
            cmd = [
                sys.executable, "-m", "agv_ga_robot.main_replay",
                "--map", map_name,
                "--genome", self.selected_genome_path
            ]
            
            subprocess.Popen(cmd, cwd=str(self.project_root))
            
            print("[Info] Replay started in separate window")
            
        except Exception as e:
            print(f"[Error] Failed to start replay: {e}")
    
    def handle_events(self) -> bool:
        """Handle input events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.current_mode is None:
                    # Mode selection
                    if self.train_button.collidepoint(event.pos):
                        self.current_mode = 'train'
                    elif self.replay_button.collidepoint(event.pos):
                        self.current_mode = 'replay'
                        self.genome_file_list = self._get_genome_files()
                        self.genome_list_scroll = 0
                
                elif self.current_mode == 'train':
                    self._handle_train_clicks(event.pos)
                
                elif self.current_mode == 'replay':
                    self._handle_replay_clicks(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                if self.active_input == 'population':
                    if event.key == pygame.K_RETURN:
                        self.active_input = None
                    elif event.key == pygame.K_BACKSPACE:
                        self.population_input = self.population_input[:-1]
                    elif event.unicode.isdigit():
                        self.population_input += event.unicode
                
                elif self.active_input == 'generations':
                    if event.key == pygame.K_RETURN:
                        self.active_input = None
                    elif event.key == pygame.K_BACKSPACE:
                        self.generations_input = self.generations_input[:-1]
                    elif event.unicode.isdigit():
                        self.generations_input += event.unicode
                
                elif event.key == pygame.K_ESCAPE:
                    if self.current_mode is not None:
                        self.current_mode = None
                        self.active_input = None
                    else:
                        return False
        
        return True
    
    def run(self):
        """Main loop."""
        running = True
        
        while running:
            running = self.handle_events()
            
            # Draw current screen
            if self.current_mode is None:
                self._draw_mode_selection()
            elif self.current_mode == 'train':
                self._draw_train_mode()
            elif self.current_mode == 'replay':
                self._draw_replay_mode()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


if __name__ == "__main__":
    gui = MainGUI()
    gui.run()
