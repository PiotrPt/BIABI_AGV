"""
Map editor - DISABLED

The map editor functionality has been disabled in this version.
Maps are now predefined as JSON files in agv_ga_robot/maps/:
  - learning_maze_simple.json
  - learning_maze_winding.json
  - learning_maze_90deg.json

To add new maps, create JSON files following the same structure as existing maps.
Map structure: metadata, robot, checkpoints, obstacles

To train or replay robots, use: python main.py
"""

if __name__ == "__main__":
    print(__doc__)

