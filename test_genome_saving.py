#!/usr/bin/env python
"""Test genome saving system"""
import os, sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from agv_ga_robot.ga.ga_trainer import GATrainer
from agv_ga_robot.utils.helpers import load_config
from agv_ga_robot.maps.map_loader import MapLoader
import numpy as np

config = load_config('agv_ga_robot/config/config.yaml')
map_data = MapLoader.load_map('agv_ga_robot/maps/learning_maze_simple.json')

trainer = GATrainer(config, map_data, output_dir='test_training_results', final_output_dir='test_training_results_final')

# Test 1
idx = trainer._find_next_final_index()
print(f'✓ Next final index: {idx}')
assert idx == 1

# Test 2
dummy = np.random.randn(100).astype(np.float32)
trainer.best_genome = dummy
trainer.save_best_genome('best_genome_gen10.pkl')
assert os.path.exists('test_training_results/best_genome_gen10.pkl')
print('✓ Gen save works')

# Test 3
trainer.save_best_genome('best_genome_final.pkl')
assert os.path.exists('test_training_results_final/best_genome_final_1.pkl')
print('✓ Final auto-increment _1')

# Test 4
trainer.save_best_genome('best_genome_final.pkl')
assert os.path.exists('test_training_results_final/best_genome_final_2.pkl')
print('✓ Final auto-increment _2')

shutil.rmtree('test_training_results', ignore_errors=True)
shutil.rmtree('test_training_results_final', ignore_errors=True)
print('\n✅ ALL GENOME SAVING TESTS PASSED')
