"""
Configuration management for Mini Metro RL project.

Provides centralized configuration loading, validation, and management
using YAML files with support for different environments and experiments.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    """Configuration for game settings."""
    map_name: str = "london"
    difficulty: str = "normal"
    real_time: bool = False
    time_step: float = 1.0
    max_episode_steps: int = 3000
    seed: Optional[int] = None
    
    # Game mechanics
    week_duration: float = 60.0
    passenger_spawn_rate: float = 1.0
    week_escalation_factor: float = 1.2
    max_station_overload_time: float = 30.0


@dataclass
class EnvironmentConfig:
    """Configuration for RL environment."""
    observation_type: str = "vector"  # "vector", "image", "graph"
    action_space_type: str = "discrete"  # "discrete", "continuous"
    reward_shaping: bool = True
    normalize_observations: bool = True
    normalize_rewards: bool = False
    
    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "passenger_delivery": 10.0,
        "passenger_satisfaction": 5.0,
        "efficiency": 3.0,
        "resource_optimization": 2.0,
        "congestion_penalty": -15.0,
        "passenger_loss_penalty": -20.0,
        "game_over_penalty": -100.0,
        "time_bonus": 1.0
    })


@dataclass
class NetworkConfig:
    """Configuration for neural network architectures."""
    # Common settings
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 512, 256])
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    
    # CNN settings (for image observations)
    conv_layers: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"filters": 32, "kernel_size": 8, "stride": 4},
        {"filters": 64, "kernel_size": 4, "stride": 2},
        {"filters": 64, "kernel_size": 3, "stride": 1}
    ])


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    # Learning parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100000
    
    # Experience replay
    buffer_size: int = 1000000
    batch_size: int = 64
    min_buffer_size: int = 50000
    
    # Target network
    target_update_frequency: int = 10000
    soft_update_tau: float = 0.005
    
    # Training
    train_frequency: int = 4
    gradient_clipping: float = 10.0
    
    # Network architecture
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Advanced features
    double_dqn: bool = True
    dueling_dqn: bool = True
    noisy_networks: bool = False
    prioritized_replay: bool = False
    
    # Multi-step learning
    n_step: int = 1


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO specific
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    
    # Network architecture
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Advanced features
    normalize_advantage: bool = True
    use_sde: bool = False
    sde_sample_freq: int = -1


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent systems."""
    agent_type: str = "dqn"  # Base agent type for each line
    num_agents: int = 3  # Maximum number of line agents
    shared_experience: bool = False
    centralized_critic: bool = False
    
    # Communication
    enable_communication: bool = False
    communication_dim: int = 64
    
    # Coordination
    coordination_graph: bool = False
    value_decomposition: str = "none"  # "none", "vdn", "qmix"


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # General training
    total_timesteps: int = 1000000
    eval_frequency: int = 10000
    eval_episodes: int = 10
    save_frequency: int = 50000
    
    # Logging
    log_frequency: int = 1000
    tensorboard_log: bool = True
    wandb_log: bool = False
    verbose: int = 1
    
    # Checkpointing
    save_path: str = "./models"
    checkpoint_frequency: int = 100000
    keep_n_checkpoints: int = 5
    
    # Early stopping
    early_stopping: bool = False
    patience: int = 10
    min_improvement: float = 0.01
    
    # Curriculum learning
    curriculum_learning: bool = False
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Hyperparameter optimization
    hyperopt: bool = False
    hyperopt_trials: int = 100
    hyperopt_sampler: str = "tpe"  # "random", "tpe"


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # TensorBoard
    tensorboard_dir: str = "./logs/tensorboard"
    log_graph: bool = True
    log_images: bool = True
    log_histograms: bool = False
    
    # Weights & Biases
    wandb_project: str = "mini-metro-rl"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Custom metrics
    custom_metrics: List[str] = field(default_factory=lambda: [
        "passenger_delivery_rate",
        "network_efficiency",
        "resource_utilization",
        "passenger_satisfaction"
    ])


@dataclass
class VisualizationConfig:
    """Configuration for visualization and rendering."""
    render_mode: str = "rgb_array"  # "human", "rgb_array", "ansi"
    fps: int = 30
    resolution: tuple = (800, 600)
    
    # Pygame rendering
    show_passenger_destinations: bool = True
    show_train_loads: bool = True
    show_station_queues: bool = True
    show_performance_metrics: bool = False
    
    # Recording
    record_videos: bool = False
    video_frequency: int = 100
    video_length: int = 1000  # steps
    
    # Real-time visualization
    realtime_plotting: bool = False
    plot_update_frequency: int = 100


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    # Sub-configurations
    game: GameConfig = field(default_factory=GameConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Experiment metadata
    experiment_name: str = "default"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config object
        """
        # Create sub-configurations
        game_config = GameConfig(**config_dict.get('game', {}))
        env_config = EnvironmentConfig(**config_dict.get('environment', {}))
        dqn_config = DQNConfig(**config_dict.get('dqn', {}))
        ppo_config = PPOConfig(**config_dict.get('ppo', {}))
        multi_agent_config = MultiAgentConfig(**config_dict.get('multi_agent', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        viz_config = VisualizationConfig(**config_dict.get('visualization', {}))
        
        # Handle nested network configs
        if 'dqn' in config_dict and 'network' in config_dict['dqn']:
            dqn_config.network = NetworkConfig(**config_dict['dqn']['network'])
        
        if 'ppo' in config_dict and 'network' in config_dict['ppo']:
            ppo_config.network = NetworkConfig(**config_dict['ppo']['network'])
        
        return cls(
            game=game_config,
            environment=env_config,
            dqn=dqn_config,
            ppo=ppo_config,
            multi_agent=multi_agent_config,
            training=training_config,
            logging=logging_config,
            visualization=viz_config,
            experiment_name=config_dict.get('experiment_name', 'default'),
            description=config_dict.get('description', ''),
            tags=config_dict.get('tags', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> 'Config':
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
            
        Returns:
            New Config object with updates applied
        """
        config_dict = self.to_dict()
        
        def recursive_update(base_dict: Dict, update_dict: Dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    recursive_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        recursive_update(config_dict, updates)
        return self.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Validate game config
        if self.game.difficulty not in ["easy", "normal", "hard", "extreme"]:
            issues.append(f"Invalid difficulty: {self.game.difficulty}")
        
        if self.game.time_step <= 0:
            issues.append("time_step must be positive")
        
        # Validate environment config
        if self.environment.observation_type not in ["vector", "image", "graph"]:
            issues.append(f"Invalid observation_type: {self.environment.observation_type}")
        
        if self.environment.action_space_type not in ["discrete", "continuous"]:
            issues.append(f"Invalid action_space_type: {self.environment.action_space_type}")
        
        # Validate learning rates
        if self.dqn.learning_rate <= 0:
            issues.append("DQN learning_rate must be positive")
        
        if self.ppo.learning_rate <= 0:
            issues.append("PPO learning_rate must be positive")
        
        # Validate training config
        if self.training.total_timesteps <= 0:
            issues.append("total_timesteps must be positive")
        
        # Validate paths
        if not Path(self.training.save_path).parent.exists():
            issues.append(f"Save path parent directory does not exist: {self.training.save_path}")
        
        return issues


def load_config(
    config_name: str = "default",
    config_dir: Union[str, Path] = None,
    overrides: Dict[str, Any] = None
) -> Config:
    """
    Load configuration with optional overrides.
    
    Args:
        config_name: Name of the configuration (without .yaml extension)
        config_dir: Directory containing configuration files
        overrides: Dictionary of configuration overrides
        
    Returns:
        Config object
    """
    if config_dir is None:
        # Default to configs directory relative to this file
        config_dir = Path(__file__).parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)
    
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}. Using defaults.")
        config = Config()
    else:
        config = Config.from_yaml(config_path)
    
    # Apply overrides
    if overrides:
        config = config.update(overrides)
    
    # Validate configuration
    issues = config.validate()
    if issues:
        logger.warning(f"Configuration validation issues: {issues}")
    
    logger.info(f"Loaded configuration: {config.experiment_name}")
    return config


def create_default_configs(config_dir: Union[str, Path]) -> None:
    """
    Create default configuration files.
    
    Args:
        config_dir: Directory to create configuration files in
    """
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Default configuration
    default_config = Config(experiment_name="default")
    default_config.to_yaml(config_dir / "default.yaml")
    
    # DQN configuration
    dqn_config = Config(
        experiment_name="dqn_baseline",
        description="Baseline DQN agent for Mini Metro",
        tags=["dqn", "baseline"]
    )
    dqn_config.to_yaml(config_dir / "dqn_config.yaml")
    
    # PPO configuration
    ppo_config = Config(
        experiment_name="ppo_baseline",
        description="Baseline PPO agent for Mini Metro",
        tags=["ppo", "baseline"]
    )
    ppo_config.to_yaml(config_dir / "ppo_config.yaml")
    
    # Game configuration with different settings
    game_config = Config(
        experiment_name="game_settings",
        description="Game-specific configuration options"
    )
    game_config.game.map_name = "paris"
    game_config.game.difficulty = "hard"
    game_config.to_yaml(config_dir / "game_config.yaml")
    
    logger.info(f"Created default configuration files in {config_dir}")