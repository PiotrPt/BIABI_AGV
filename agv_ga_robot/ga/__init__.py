"""GA module - genetic algorithm trainer and fitness functions."""

from .ga_trainer import GATrainer
from .fitness import calculate_fitness

__all__ = ["GATrainer", "calculate_fitness"]
