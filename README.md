# AGV Genetic Algorithm Robot Training System

Complete implementation of an Automated Guided Vehicle (AGV) trained with genetic algorithms to navigate mazes and reach checkpoints.

## Overview

This project combines:
- **Robot Physics**: Differential drive kinematics, distance sensors, collision detection
- **Genetic Algorithm**: PyGAD-based population-based optimization
- **Neural Network**: Feedforward network (16→64→32→2) for motor control
- **Real-time Visualization**: Single-window live display with overlayed trajectories
- **Map Editor**: Interactive Pygame-based obstacle design

## Architecture

### Robot System
- **Differential Drive**: Two independent wheel motors (-1 to +1 speed range)
- **Sensors**: 3 distance sensors (front 0°, left -60°, right +60°) using raycasting
- **Collision Detection**: Ray-to-line-segment distance with obstacle width
- **Observation Space**: 16D vector containing sensor readings and checkpoint information

### Neural Network
- **Architecture**: 16 → 64 → 32 → 2 (input → hidden1 → hidden2 → output)
- **Activation**: Tanh (hidden + output layers)
- **Parameters**: ~1250 total (genome size for GA)
- **Output**: Motor commands [left_motor, right_motor] in [-1, 1]

### Genetic Algorithm
- **Population**: 180 individuals per generation
- **Generations**: 250
- **Selection**: Tournament (size 5)
- **Crossover**: Single-point
- **Mutation**: 20% per gene
- **Fitness Evaluation**: Each genome evaluated 1 time (single evaluation)

### Reward Function
Multi-component reward encouraging checkpoint collection and obstacle avoidance:
```
reward = 15000×checkpoints_reached + 120×distance_improvement + 80×forward_bonus
         + 0.5×speed_bonus - 30×collisions - 200×stuck - 10×(no_progress)
         - 0.2×idle_rotation - 0.001×sharp_rotation - 3000×timeout
```

Reward components:
- **checkpoint_reached**: 15000 per checkpoint (higher multiplier for later checkpoints)
- **distance_improvement**: 120 for each pixel closer to next checkpoint
- **forward_bonus**: 80 when moving mostly straight
- **collision_penalty**: -30 per collision
- **progress_penalty**: -10 after 50 steps without progress
- **stuck_penalty**: -200 when robot is immobile
- **idle_rotation_penalty**: -0.2 for idle turning in place
- **sharp_rotation_penalty**: -0.001 for sharp turns while moving
- **timeout_penalty**: -3000 if episode timeout reached

### Observation Space (16D)
```
[0-2]   : 3 sensor readings (normalized 0-1)
[3-14]  : 3 checkpoints × [distance_norm, sin(angle), cos(angle), visited_flag]
[15]    : Priority signal (for checkpoint ordering)
```

## Project Structure

```
agv_ga_robot/
├── config/
│   └── config.yaml                 # All tunable parameters
├── maps/
│   └── learning_maze_simple.json   # Default training map
├── robot/
│   ├── robot_agent.py              # Differential drive physics
│   ├── sensors.py                  # 3D raycasting
│   └── neural_network.py           # Genome-based NN
├── env/
│   └── agv_env.py                  # Gymnasium-compliant environment
├── ga/
│   ├── fitness.py                  # Fitness calculation
│   └── ga_trainer.py               # PyGAD wrapper
├── ui/
│   ├── visualization_simple.py      # Real-time single-window display
│   └── map_editor.py               # Interactive obstacle editor
├── utils/
│   ├── helpers.py                  # Config loading, normalization
│   └── ...
├── maps/
│   └── map_loader.py               # JSON map I/O
├── main_train.py                   # GA training with visualization
├── main_replay.py                  # Replay trained genome
├── main_editor.py                  # Launch map editor
├── test_phase1.py                  # Unit tests (robot, sensors, NN, env)
└── test_phase2.py                  # Integration tests (GA, training)
```

## Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.14.4)
- Virtual environment (recommended)

### Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install gymnasium numpy pyyaml pygad pygame-ce
   ```

   Specific versions used:
   - gymnasium==1.3.0
   - numpy==2.4.4
   - pyyaml
   - pygad==3.6.0
  - pygame-ce (2.5.7+) - community edition with Python 3.14 support

## Commands

Run these common commands after activating the virtual environment. Paths are repository-relative.

- Activate (Windows):
```powershell
.venv\Scripts\activate
```

- Activate (Linux/macOS):
```bash
source .venv/bin/activate
```

- Install dependencies (recommended):
```bash
pip install -r requirements.txt
# or: pip install gymnasium numpy pyyaml pygad pygame-ce
```

- Train with visualization (default config/map):
```bash
python -m agv_ga_robot.main_train
```

- Train with custom map:
```bash
python -m agv_ga_robot.main_train --map learning_maze_simple --config agv_ga_robot/config/config.yaml
```

- Train with custom population and generations:
```bash
python -m agv_ga_robot.main_train --map learning_maze_simple --population 80 --generations 150
```

- Replay a trained genome (automatically finds latest):
```bash
python -m agv_ga_robot.main_replay
```

- Replay specific genome:
```bash
python -m agv_ga_robot.main_replay --genome training_results_final/best_genome_final_0.pkl --map learning_maze_simple
```

- Launch map editor:
```bash
python -m agv_ga_robot.main_editor
```

- Run tests (from repo root):
```bash
python agv_ga_robot/test_phase1.py
python agv_ga_robot/test_phase2.py
```

## Usage

All operational commands are consolidated in the **Commands** section above. Activate the virtualenv and run the appropriate command from that list. Use `--help` on each script for additional flags and options.

## Configuration

All parameters are in `config/config.yaml`:

```yaml
GA_PARAMS:
  population_size: 180
  num_generations: 250
  mutation_percent: 0.20
  fitness_evaluations: 1
  crossover_type: single_point
  parent_selection: tournament
  tournament_size: 5

NETWORK_ARCH: [16, 64, 32, 2]

REWARD_WEIGHTS:
  checkpoint_reached: 15000
  distance_improvement: 120.0
  forward_bonus: 80.0
  collision_penalty: -30.0
  progress_penalty: -10.0
  stuck_penalty: -200.0
  idle_rotation_penalty: -0.2
  sharp_rotation_penalty: -0.001
  timeout_penalty: -3000.0
  speed_bonus: 0.5

EPISODE_STEPS: 500

ROBOT:
  width: 30
  height: 20
  max_speed: 1.0
  stuck_threshold: 0.05
  sensor_angles: [-60, 0, 60]
  sensor_max_distance: 50

SIMULATION:
  dt: 0.016

CHECKPOINT_MODE: "ordered"
CHECKPOINT_RADIUS: 50
```

## Visualization Features

### Single Window Display (Pygame-based)
- **Real-time multi-robot display**: Leader (green) + Top 3 from current generation (blue/cyan/magenta)
- **Trajectories**: Lines showing each robot's path through the map
- **Sensor visualization**: 3 distance sensor rays (-60°, 0°, +60°) from each robot
- **Target arrow**: Direction indicator to next unvisited checkpoint

### Legend and Generation Info
- **Generation stats**: Current generation, best fitness, population average fitness
- **Trajectory colors**: 
  - Green = Leader (best from previous generation)
  - Blue = Rank 1 (best from current generation)
  - Cyan = Rank 2 (2nd best from current generation)  
  - Magenta = Rank 3 (3rd best from current generation)

### Display Elements
- **Gray lines**: Obstacles
- **Colored lines**: Robot trajectories (green/blue/cyan/magenta)
- **Colored circles**: Current robot positions
- **Sensor rays**: Distance sensor readings (lighter shade of robot color)
- **Target arrows**: Direction to next checkpoint
- **Orange circles**: Unvisited checkpoints
- **Green circles**: Visited checkpoints

## Key Findings

### Reward Function Design
Initial sparse reward (+50 only at checkpoint) caused robots to wait. Added **distance_improvement** (+0.5 per pixel closer) provides continuous guidance signal.

### Angle Representation
Linear encoding of angles (-1 to +1) causes aliasing (359° ≈ 1°). Solution: **sin(angle) + cos(angle)** representation - naturally periodic, no aliasing.

### Fitness Stability
Single evaluation per genome allows "lucky" genomes. Solution: **3× evaluation** with different random seeds, average fitness used for selection.

### Network Architecture
16→32→16→2 chosen empirically:
- 16D input (3 sensors + 12 checkpoint features + 1 priority)
- 32 hidden neurons (captures sensor fusion + navigation logic)
- 16 hidden neurons (refinement layer)
- 2D output (left/right motor commands)

## Training Performance

Typical GA training (250 gen × 180 pop):
- **Generation 1**: Best fitness ~190000-200000
- **Generation 50**: Best fitness ~210000-220000
- **Generation 100**: Best fitness ~220000-240000
- **Generation 250**: Best fitness ~240000+
- **Total time**: ~3-6 hours (depends on hardware, episode length)

Successfully trained robots exhibit:
- Navigation through complex mazes
- Checkpoint collection in order
- Obstacle avoidance
- Efficient path planning
- Forward-biased movement (reduced idle spinning)

## Extending the System

### Add More Checkpoints
Edit `maps/learning_maze_simple.json` to add more checkpoint objects

### Modify Map Layout
Use `main_editor.py` to interactively design new training environments

### Adjust GA Parameters
Edit `config/config.yaml` to change population size, mutation rate, etc.

### Customize Reward Function
Modify `ga/fitness.py` and `REWARD_WEIGHTS` in config

### Extend Network Architecture
Change `NETWORK_ARCH` in config or modify `robot/neural_network.py`

## Troubleshooting

### Import Errors
Ensure you're running from the correct directory, have activated the virtual environment, and are using the venv Python interpreter:
```bash
cd d:\BIAI2
.venv\Scripts\activate
```

Or use the venv Python directly without activation:
```bash
.venv\Scripts\python -m agv_ga_robot.main_editor
```

### Pygame Display Issues
On headless systems, use `--no-viz` flag or run with `DISPLAY=:0` on Linux

**Python 3.14 Note**: Standard `pygame` package doesn't support Python 3.14. Use `pygame-ce` (community edition) instead:
```bash
pip install pygame-ce
```

### Out of Memory
Reduce `population_size` or `fitness_evaluations` in config if training uses too much RAM

### Slow Training
- Reduce `EPISODE_STEPS` for faster evaluation
- Decrease `fitness_evaluations` (trades stability for speed)
- Use `--no-viz` to save GPU resources

## References

- **PyGAD**: https://github.com/ahmedfayed/pygad
- **Gymnasium**: https://gymnasium.farama.org/
- **Differential Drive**: https://en.wikipedia.org/wiki/Differential_wheeled_robot
- **Raycasting**: Game engine collision detection technique

## License

Open source - feel free to use and modify for educational/research purposes.

## Contact

For questions or issues, refer to the inline code documentation or contact the development team.
