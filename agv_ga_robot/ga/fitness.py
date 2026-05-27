"""Fitness function - Calculate episode fitness for GA."""

from typing import Dict, Any


def calculate_fitness(episode_data: Dict[str, Any], reward_weights: Dict[str, float]) -> float:
    """
    Calculate final fitness score for an episode.
    
    The fitness is based on the accumulated reward during the episode,
    which already accounts for all components (checkpoints, distance, speed, etc).
    
    We add an offset to ensure fitness is always positive (PyGAD handles this better),
    while maintaining clear distinction between good and bad behavior.
    
    Args:
        episode_data: Dictionary with episode statistics from AGVEnvironment
        reward_weights: Dictionary with reward weights from config
    
    Returns:
        Final fitness score (always positive with offset)
    """
    # Use accumulated episode reward directly
    total_reward = episode_data.get('total_reward', 0.0)
    
    # Add offset to ensure positive fitness (PyGAD works better with positive values)
    # Offset: 200,000 - ensures massive spread for selection:
    # - No checkpoints, deep progress_penalty: ~-10,000 → fitness ~190,000 (very bad, dies)
    # - 1 checkpoint, some movement: ~+5,000 → fitness ~205,000 (mediocre)
    # - 2 checkpoints, distance: ~+20,000 → fitness ~220,000 (good)
    # - 3 checkpoints: ~+30,000 → fitness ~230,000 (excellent)
    FITNESS_OFFSET = 200000.0
    fitness = total_reward + FITNESS_OFFSET
    
    # Ensure minimum fitness of 0
    return max(0.0, fitness)


def get_episode_stats_from_env(env) -> Dict[str, Any]:
    """
    Extract episode statistics from environment after episode.
    
    Args:
        env: AGVEnvironment instance
    
    Returns:
        Dictionary with episode statistics
    """
    summary = env.episode_data.get_summary()
    return {
        'checkpoints_visited': summary.get('checkpoints_visited', 0),
        'collision_count': summary.get('collision_count', 0),
        'total_distance_improvement': summary.get('total_distance_improvement', 0.0),
        'total_speed': summary.get('total_speed', 0.0),
        'stuck_count': summary.get('stuck_count', 0),
        'steps': summary.get('steps', 0),
        'total_reward': summary.get('total_reward', 0.0),
        'avg_speed': summary.get('avg_speed', 0.0),
    }
