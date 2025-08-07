"""
Mini Metro Reinforcement Learning Project

A comprehensive implementation of the Mini Metro game with advanced reinforcement learning agents.
Features include accurate game mechanics, multiple RL algorithms (DQN, PPO, Multi-agent),
extensive visualization, and production-ready deployment capabilities.
"""

__version__ = "1.0.0"
__author__ = "Mini Metro RL Team"
__email__ = "contact@minimetro-rl.com"

# Core imports for easy access
from mini_metro_rl.src.game.mini_metro_game import MiniMetroGame
from mini_metro_rl.src.environment.metro_env import MetroEnvironment
from mini_metro_rl.src.agents.dqn_agent import DQNAgent
from mini_metro_rl.src.agents.ppo_agent import PPOAgent

__all__ = [
    "MiniMetroGame",
    "MetroEnvironment", 
    "DQNAgent",
    "PPOAgent",
]