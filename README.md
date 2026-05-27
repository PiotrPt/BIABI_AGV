# AGV Genetic Algorithm Robot Training System

Complete implementation of an Automated Guided Vehicle (AGV) trained with genetic algorithms to navigate mazes and reach checkpoints.

## Overview

This project combines:
- **Robot Physics**: Differential drive kinematics, distance sensors, collision detection
- **Genetic Algorithm**: PyGAD-based population-based optimization
- **Neural Network**: Feedforward network (16→32→16→2) for motor control
- **Real-time Visualization**: 4-window layout for training monitoring
- **Map Editor**: Interactive Pygame-based obstacle design

## Architecture

### Robot System
- **Differential Drive**: Two independent wheel motors (-1 to +1 speed range)
- **Sensors**: 3 distance sensors (front 0°, left -60°, right +60°) using raycasting
- **Collision Detection**: Ray-to-line-segment distance with obstacle width
- **Observation Space**: 16D vector containing sensor readings and checkpoint information

### Neural Network
- **Architecture**: 16 → 32 → 16 → 2 (input → hidden1 → hidden2 → output)
- **Activation**: Tanh (hidden + output layers)
- **Parameters**: 1106 total (genome size for GA)
- **Output**: Motor commands [left_motor, right_motor] in [-1, 1]

### Genetic Algorithm
- **Population**: 60 individuals per generation
- **Generations**: 100
- **Selection**: Tournament (size 3)
- **Crossover**: Single-point
- **Mutation**: 15% per gene
- **Fitness Evaluation**: Each genome evaluated 3 times with different seeds for stability

### Reward Function
Multi-component reward encouraging checkpoint collection and obstacle avoidance:
```
reward = 50×checkpoints_reached - 10×collisions + 0.5×distance_improvement 
         + 0.1×avg_speed - 0.02×steps - 1.0×stuck_count
```

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
│   ├── fitness.py                  # Multi-component reward calculation
│   └── ga_trainer.py               # PyGAD wrapper
├── ui/
│   ├── visualization.py            # Real-time 4-window display
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

## Usage

**Important**: Always activate the virtual environment before running commands:
```bash
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 1. Training with Visualization

Run GA training with real-time 4-window visualization:

```bash
python -m agv_ga_robot.main_train \
    --config agv_ga_robot/config/config.yaml \
    --map agv_ga_robot/maps/learning_maze_simple.json \
    --output training_results
```

**Controls:**
- Close window or press ESC to stop training
- Results saved to `training_results/`

### 2. Replay Trained Genome

Visualize a trained genome's performance:

```bash
python -m agv_ga_robot.main_replay \
    --config agv_ga_robot/config/config.yaml \
    --map agv_ga_robot/maps/learning_maze_simple.json \
    --genome training_results/best_genome.pkl
```

### 3. Interactive Map Editor

Create custom obstacle layouts:

```bash
python -m agv_ga_robot.main_editor \
    --config agv_ga_robot/config/config.yaml \
    --output custom_map.json
```

**Controls:**
- **Draw**: Draw obstacles by dragging mouse (LMB)
- **CP**: Place checkpoints (click on canvas)
- **Start**: Set robot start position (click on canvas)
- **Undo**: Undo last action
- **Clear**: Clear all elements
- **Save**: Save map to JSON
- **Load**: Load map from JSON
- **Ctrl+Z**: Undo shortcut

### 4. Run Tests

Unit tests (Phase 1 - robot, sensors, NN, env):
```bash
cd d:\BIAI2\agv_ga_robot
d:\BIAI2\.venv\Scripts\python test_phase1.py
```

Integration tests (Phase 2 - GA, training):
```bash
cd d:\BIAI2\agv_ga_robot
d:\BIAI2\.venv\Scripts\python test_phase2.py
```

## Configuration

All parameters are in `config/config.yaml`:

```yaml
GA_PARAMS:
  population_size: 60
  num_generations: 100
  mutation_percent: 0.15
  fitness_evaluations: 3
  crossover_type: single_point
  parent_selection: tournament
  tournament_size: 3

NETWORK_ARCH: [16, 32, 16, 2]

REWARD_WEIGHTS:
  checkpoint_reached: 50
  collision_penalty: -10
  distance_improvement: 0.5
  speed_bonus: 0.1
  time_penalty: -0.02
  stuck_penalty: -1.0

EPISODE_STEPS: 2500

ROBOT:
  width: 20
  height: 20
  max_speed: 1.0
  axle_distance: 15

SIMULATION:
  dt: 0.016

CHECKPOINT_RADIUS: 50

SENSOR:
  angles: [-60, 0, 60]
  max_distance: 200
```

## Visualization Features

### Single Window Layout (Pygame-based)
- **Real-time 4-robot display**: Leader (green) + Top 3 from current generation (blue/cyan/magenta)
- **Trajectories**: Lines showing each robot's path through the map
- **Sensor visualization**: 3 distance sensor rays (-60°, 0°, +60°) from each robot
- **Target arrow**: Direction indicator to next unvisited checkpoint

### Legend and Stats Panel
- **Generation info**: Current generation, best fitness, population average
- **Reward breakdown**: Per-robot reward component breakdown showing:
  - Checkpoint rewards (cumulative)
  - Distance improvement bonus
  - Forward movement bonus
  - Collision penalties
  - Idle rotation penalties
  - Sharp rotation penalties
  - Stuck penalties

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

Typical GA training (100 gen × 60 pop):
- **Generation 1**: Best fitness ~100-200
- **Generation 50**: Best fitness ~500-800
- **Generation 100**: Best fitness ~1000+
- **Total time**: ~2-4 hours (depends on hardware, episode length)

Successfully trained robots exhibit:
- Navigation through complex mazes
- Checkpoint collection in order
- Obstacle avoidance
- Efficient path planning

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
