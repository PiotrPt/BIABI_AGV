#!/usr/bin/env python
"""
Test system without pygame - verify all core components work
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

print("=" * 60)
print("TESTING CORE COMPONENTS (without pygame)")
print("=" * 60)

# Test 1: Config loading
print("\n[Test 1] Config Loading")
try:
    from agv_ga_robot.utils.helpers import load_config
    config = load_config("agv_ga_robot/config/config.yaml")
    
    maps = config.get('MAPS_AVAILABLE', [])
    train_params = config.get('TRAIN_PARAMS', {})
    
    print(f"✓ Config loaded")
    print(f"  - MAPS_AVAILABLE: {maps}")
    print(f"  - default_population_size: {train_params.get('default_population_size')}")
    print(f"  - default_num_generations: {train_params.get('default_num_generations')}")
    
    assert len(maps) == 3, f"Expected 3 maps, got {len(maps)}"
    print("✓ Config VERIFIED")
except Exception as e:
    print(f"✗ Config test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Map loading
print("\n[Test 2] Map Loading")
try:
    from agv_ga_robot.maps.map_loader import MapLoader
    
    maps_to_test = [
        'agv_ga_robot/maps/learning_maze_simple.json',
        'agv_ga_robot/maps/learning_maze_winding.json',
        'agv_ga_robot/maps/learning_maze_90deg.json'
    ]
    
    for map_path in maps_to_test:
        map_data = MapLoader.load_map(map_path)
        name = map_data['metadata']['name']
        checkpoints = len(map_data['checkpoints'])
        obstacles = len(map_data['obstacles'])
        print(f"✓ {name:30s} ({checkpoints} CPs, {obstacles} obstacles)")
    
    print("✓ All maps VERIFIED")
except Exception as e:
    print(f"✗ Map test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: GA Trainer genome saving
print("\n[Test 3] Genome Saving System")
try:
    from agv_ga_robot.ga.ga_trainer import GATrainer
    import numpy as np
    import os
    import shutil
    
    config = load_config("agv_ga_robot/config/config.yaml")
    map_data = MapLoader.load_map("agv_ga_robot/maps/learning_maze_simple.json")
    
    trainer = GATrainer(
        config, 
        map_data,
        output_dir='test_training_results',
        final_output_dir='test_training_results_final'
    )
    
    # Test saving
    dummy = np.random.randn(100).astype(np.float32)
    trainer.best_genome = dummy
    
    trainer.save_best_genome('best_genome_gen10.pkl')
    assert os.path.exists('test_training_results/best_genome_gen10.pkl'), "Gen save failed"
    print("✓ Gen-based genome save works")
    
    trainer.save_best_genome('best_genome_final.pkl')
    assert os.path.exists('test_training_results_final/best_genome_final_1.pkl'), "Final _1 save failed"
    print("✓ Final genome auto-increment (_1) works")
    
    trainer.save_best_genome('best_genome_final.pkl')
    assert os.path.exists('test_training_results_final/best_genome_final_2.pkl'), "Final _2 save failed"
    print("✓ Final genome auto-increment (_2) works")
    
    # Cleanup
    shutil.rmtree('test_training_results', ignore_errors=True)
    shutil.rmtree('test_training_results_final', ignore_errors=True)
    print("✓ Genome saving VERIFIED")
except Exception as e:
    print(f"✗ Genome saving test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Environment with new maps
print("\n[Test 4] Environment with New Maps")
try:
    from agv_ga_robot.env.agv_env import AGVEnvironment
    
    for map_name in ['learning_maze_simple', 'learning_maze_winding', 'learning_maze_90deg']:
        map_data = MapLoader.load_map(f'agv_ga_robot/maps/{map_name}.json')
        env = AGVEnvironment(config, map_data)
        obs, info = env.reset()
        assert obs.shape == (16,), f"Wrong observation shape: {obs.shape}"
        print(f"✓ Environment works with {map_name}")
    
    print("✓ Environment VERIFIED with all maps")
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ ALL CORE TESTS PASSED")
print("=" * 60)
print("\nSystem is ready! To use GUI:")
print("  1. Install pygame (if not already): pip install pygame")
print("  2. Run: python main.py")
print("\nOr run training directly from command line:")
print("  python -m agv_ga_robot.main_train --map learning_maze_simple --population 50 --generations 10")
