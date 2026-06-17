#!/usr/bin/env python
"""Test config"""
from agv_ga_robot.utils.helpers import load_config

config = load_config('agv_ga_robot/config/config.yaml')

print('✓ Config loaded')
maps = config.get('MAPS_AVAILABLE')
print(f'✓ MAPS_AVAILABLE: {maps}')

train_params = config.get('TRAIN_PARAMS', {})
print(f'✓ TRAIN_PARAMS:')
print(f'  - default_population_size: {train_params.get("default_population_size", "NOT FOUND")}')
print(f'  - default_num_generations: {train_params.get("default_num_generations", "NOT FOUND")}')

assert maps is not None, "MAPS_AVAILABLE not found in config"
assert len(maps) == 3, f"Expected 3 maps, got {len(maps)}"
assert train_params.get('default_population_size') is not None, "default_population_size not found"

print('✅ CONFIG STRUCTURE VERIFIED')
