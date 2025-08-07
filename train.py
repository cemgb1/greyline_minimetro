#!/usr/bin/env python3
"""
Training script for Mini Metro RL agents.

Dedicated training script with advanced features like hyperparameter optimization,
curriculum learning, and comprehensive experiment tracking.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import optuna
from datetime import datetime
import yaml

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mini_metro_rl.src.utils.config import load_config, Config
from mini_metro_rl.src.utils.helpers import set_random_seeds, Timer, PerformanceMonitor
from mini_metro_rl.src.environment.metro_env import MetroEnvironment, EnvironmentConfig
from mini_metro_rl.src.agents.dqn_agent import DQNAgent
from mini_metro_rl.src.agents.ppo_agent import PPOAgent
from mini_metro_rl.src.agents.multi_agent import MultiAgent
from mini_metro_rl.src.visualization.tensorboard_logger import TensorboardLogger

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Manages the training process with advanced features.
    
    Features:
    - Hyperparameter optimization
    - Curriculum learning
    - Early stopping
    - Model checkpointing
    - Comprehensive logging
    """
    
    def __init__(self, config: Config):
        """
        Initialize training manager.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        
        # Setup paths
        self.model_dir = Path(config.training.save_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.logging.tensorboard_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_reward = float('-inf')
        self.best_model_path = None
        self.early_stop_counter = 0
        
        # Curriculum learning
        self.current_curriculum_stage = 0
        
        logger.info(f"Initialized training manager for {config.experiment_name}")
    
    def create_environment(self, stage_config: Optional[Dict] = None) -> MetroEnvironment:
        """
        Create environment with optional curriculum stage modifications.
        
        Args:
            stage_config: Optional curriculum stage configuration
            
        Returns:
            Configured environment
        """
        # Base config
        env_config = EnvironmentConfig(
            map_name=self.config.game.map_name,
            difficulty=self.config.game.difficulty,
            max_episode_steps=self.config.game.max_episode_steps,
            observation_type=self.config.environment.observation_type,
            action_space_type=self.config.environment.action_space_type,
            reward_weights=self.config.environment.reward_weights,
            real_time=self.config.game.real_time,
            time_step=self.config.game.time_step,
            seed=self.config.game.seed
        )
        
        # Apply curriculum modifications
        if stage_config:
            if 'difficulty' in stage_config:
                env_config.difficulty = stage_config['difficulty']
            if 'max_stations' in stage_config:
                # Would need to modify game config
                pass
        
        return MetroEnvironment(env_config)
    
    def create_agent(self, env: MetroEnvironment, trial: Optional[optuna.Trial] = None) -> Any:
        """
        Create agent with optional hyperparameter optimization.
        
        Args:
            env: Environment instance
            trial: Optuna trial for hyperparameter optimization
            
        Returns:
            Configured agent
        """
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.nvec[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        agent_type = self.config.multi_agent.agent_type
        
        if trial is not None:
            # Hyperparameter optimization
            if agent_type == "dqn":
                config = self._optimize_dqn_hyperparameters(trial)
            elif agent_type == "ppo":
                config = self._optimize_ppo_hyperparameters(trial)
            else:
                config = self.config.dqn if agent_type == "dqn" else self.config.ppo
        else:
            config = self.config.dqn if agent_type == "dqn" else self.config.ppo
        
        # Create agent
        if agent_type == "dqn":
            return DQNAgent(state_dim, action_dim, config, device)
        elif agent_type == "ppo":
            return PPOAgent(state_dim, action_dim, config, device)
        elif agent_type == "multi":
            agent_configs = {"dqn": self.config.dqn, "ppo": self.config.ppo}
            return MultiAgent(state_dim, action_dim, self.config.multi_agent, agent_configs, device)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _optimize_dqn_hyperparameters(self, trial: optuna.Trial) -> Any:
        """Optimize DQN hyperparameters."""
        from copy import deepcopy
        config = deepcopy(self.config.dqn)
        
        # Suggest hyperparameters
        config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        config.gamma = trial.suggest_float("gamma", 0.95, 0.999)
        config.epsilon_decay_steps = trial.suggest_int("epsilon_decay_steps", 50000, 200000)
        config.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        config.target_update_frequency = trial.suggest_int("target_update_frequency", 5000, 20000)
        
        # Network architecture
        config.network.hidden_sizes = [
            trial.suggest_categorical("hidden_size_1", [256, 512, 768, 1024]),
            trial.suggest_categorical("hidden_size_2", [256, 512, 768]),
            trial.suggest_categorical("hidden_size_3", [128, 256, 512])
        ]
        
        return config
    
    def _optimize_ppo_hyperparameters(self, trial: optuna.Trial) -> Any:
        """Optimize PPO hyperparameters."""
        from copy import deepcopy
        config = deepcopy(self.config.ppo)
        
        # Suggest hyperparameters
        config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        config.gamma = trial.suggest_float("gamma", 0.95, 0.999)
        config.gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        config.clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        config.entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.1, log=True)
        config.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        config.n_epochs = trial.suggest_int("n_epochs", 5, 20)
        
        return config
    
    def train_single_run(
        self,
        trial: Optional[optuna.Trial] = None,
        stage_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train agent for a single run.
        
        Args:
            trial: Optuna trial for hyperparameter optimization
            stage_config: Curriculum stage configuration
            
        Returns:
            Training metrics
        """
        # Create environment and agent
        env = self.create_environment(stage_config)
        agent = self.create_agent(env, trial)
        
        # Setup logging
        run_name = f"{self.config.experiment_name}"
        if trial:
            run_name += f"_trial_{trial.number}"
        if stage_config:
            run_name += f"_stage_{stage_config.get('stage', 0)}"
        
        tb_logger = TensorboardLogger(
            str(self.log_dir / run_name),
            self.config.logging.log_graph,
            self.config.logging.log_images,
            self.config.logging.log_histograms
        )
        
        # Training loop
        total_steps = 0
        episode = 0
        episode_rewards = []
        
        target_steps = stage_config.get('steps', self.config.training.total_timesteps) if stage_config else self.config.training.total_timesteps
        
        logger.info(f"Starting training run: {run_name}")
        
        with Timer(f"Training {run_name}", log_result=True):
            while total_steps < target_steps:
                episode += 1
                episode_reward = 0
                episode_steps = 0
                
                # Reset environment
                obs, info = env.reset()
                done = False
                
                while not done and episode_steps < self.config.game.max_episode_steps:
                    # Get action
                    action_mask = env.get_action_mask()
                    
                    if hasattr(agent, 'act'):
                        if isinstance(agent, PPOAgent):
                            action, log_prob, value = agent.act(obs, action_mask)
                            action_info = {'action': action, 'log_prob': log_prob, 'value': value}
                        else:
                            action = agent.act(obs, action_mask)
                            action_info = {'action': action}
                    else:
                        # Multi-agent
                        actions = agent.act({0: obs}, {0: action_mask} if action_mask is not None else None)
                        action_info = actions.get(0, {'action': 0})
                        action = action_info['action']
                    
                    # Take step
                    next_obs, reward, terminated, truncated, step_info = env.step(action)
                    done = terminated or truncated
                    
                    # Update agent
                    if hasattr(agent, 'step'):
                        if isinstance(agent, PPOAgent):
                            training_metrics = agent.step(
                                obs, action, reward, next_obs, done,
                                action_info.get('log_prob', 0.0),
                                action_info.get('value', 0.0)
                            )
                            if training_metrics:
                                tb_logger.log_agent_metrics(total_steps, training_metrics, "ppo")
                        else:
                            agent.step(obs, action, reward, next_obs, done)
                    else:
                        # Multi-agent
                        agent.step({0: obs}, {0: action_info}, {0: reward}, {0: next_obs}, {0: done})
                    
                    # Update tracking
                    obs = next_obs
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1
                    
                    # Logging
                    if total_steps % self.config.training.log_frequency == 0:
                        tb_logger.log_scalar("training/episode_reward", episode_reward, total_steps)
                        tb_logger.log_scalar("training/total_steps", total_steps, total_steps)
                        
                        # Log reward breakdown
                        if 'reward_breakdown' in step_info:
                            tb_logger.log_reward_breakdown(total_steps, step_info['reward_breakdown'])
                        
                        # Log game state
                        game_state = env.game.get_detailed_state()
                        tb_logger.log_game_state_metrics(total_steps, game_state)
                        
                        # Log agent metrics
                        agent_metrics = agent.get_metrics()
                        tb_logger.log_agent_metrics(total_steps, agent_metrics)
                
                # End episode
                if hasattr(agent, 'end_episode'):
                    agent.end_episode()
                
                episode_rewards.append(episode_reward)
                
                # Performance monitoring
                self.performance_monitor.record(
                    episode_reward=episode_reward,
                    episode_steps=episode_steps,
                    total_steps=total_steps
                )
                
                # Logging and checkpointing
                if episode % 10 == 0:
                    recent_avg_reward = np.mean(episode_rewards[-10:])
                    logger.info(f"Episode {episode}: reward={episode_reward:.2f}, "
                               f"avg_reward={recent_avg_reward:.2f}, steps={total_steps}")
                    
                    tb_logger.log_scalar("training/avg_reward_10", recent_avg_reward, total_steps)
                
                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.best_model_path = str(self.model_dir / f"best_{run_name}.pt")
                    agent.save(self.best_model_path)
                    logger.info(f"Saved best model with reward {self.best_reward:.2f}")
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                # Early stopping
                if (self.config.training.early_stopping and 
                    self.early_stop_counter >= self.config.training.patience):
                    logger.info(f"Early stopping triggered after {self.early_stop_counter} episodes")
                    break
                
                # Checkpointing
                if episode % (self.config.training.checkpoint_frequency // 1000) == 0:
                    checkpoint_path = str(self.model_dir / f"checkpoint_{run_name}_episode_{episode}.pt")
                    agent.save(checkpoint_path)
        
        # Cleanup
        tb_logger.close()
        
        # Calculate final metrics
        final_metrics = {
            'total_episodes': episode,
            'total_steps': total_steps,
            'best_reward': self.best_reward,
            'mean_reward': np.mean(episode_rewards),
            'final_reward': episode_rewards[-1] if episode_rewards else 0,
            'performance_summary': self.performance_monitor.summary()
        }
        
        return final_metrics
    
    def train_with_curriculum(self) -> Dict[str, Any]:
        """
        Train with curriculum learning.
        
        Returns:
            Training metrics
        """
        if not self.config.training.curriculum_learning:
            return self.train_single_run()
        
        logger.info("Starting curriculum learning")
        
        all_metrics = {}
        
        for stage_idx, stage_config in enumerate(self.config.training.curriculum_stages):
            logger.info(f"Starting curriculum stage {stage_idx + 1}: {stage_config}")
            
            stage_metrics = self.train_single_run(stage_config=stage_config)
            all_metrics[f"stage_{stage_idx + 1}"] = stage_metrics
            
            logger.info(f"Completed stage {stage_idx + 1}: {stage_metrics}")
        
        return all_metrics
    
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Returns:
            Optimization results
        """
        if not self.config.training.hyperopt:
            logger.info("Hyperparameter optimization disabled")
            return {}
        
        logger.info(f"Starting hyperparameter optimization with {self.config.training.hyperopt_trials} trials")
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function for optimization."""
            try:
                metrics = self.train_single_run(trial=trial)
                return metrics['mean_reward']
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return float('-inf')
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler() if self.config.training.hyperopt_sampler == 'tpe' else optuna.samplers.RandomSampler()
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.config.training.hyperopt_trials)
        
        # Results
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        # Save results
        results_path = self.model_dir / "hyperopt_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Hyperparameter optimization completed. Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Mini Metro RL Training")
    
    # Configuration
    parser.add_argument("--config", type=str, default="dqn_config",
                       help="Configuration file name")
    parser.add_argument("--config-dir", type=str, default=None,
                       help="Configuration directory")
    
    # Training options
    parser.add_argument("--hyperopt", action="store_true",
                       help="Run hyperparameter optimization")
    parser.add_argument("--curriculum", action="store_true",
                       help="Use curriculum learning")
    parser.add_argument("--trials", type=int, help="Number of hyperopt trials")
    parser.add_argument("--steps", type=int, help="Number of training steps")
    
    # Output options
    parser.add_argument("--save-dir", type=str, help="Directory to save models")
    parser.add_argument("--log-dir", type=str, help="Directory for logs")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    
    # Other options
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO")
    
    args = parser.parse_args()
    
    # Load configuration
    config_overrides = {}
    if args.save_dir:
        config_overrides['training'] = {'save_path': args.save_dir}
    if args.log_dir:
        config_overrides['logging'] = {'tensorboard_dir': args.log_dir}
    if args.experiment_name:
        config_overrides['experiment_name'] = args.experiment_name
    if args.seed:
        config_overrides['game'] = {'seed': args.seed}
    if args.steps:
        config_overrides['training'] = {'total_timesteps': args.steps}
    if args.trials:
        config_overrides['training'] = {'hyperopt_trials': args.trials}
    if args.hyperopt:
        config_overrides['training'] = {'hyperopt': True}
    if args.curriculum:
        config_overrides['training'] = {'curriculum_learning': True}
    
    config = load_config(args.config, args.config_dir, config_overrides)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=config.logging.log_format
    )
    
    # Set random seeds
    if config.game.seed:
        set_random_seeds(config.game.seed)
    
    # Set device
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Starting training: {config.experiment_name}")
    logger.info(f"Device: {device}")
    
    try:
        # Create training manager
        trainer = TrainingManager(config)
        
        # Run training
        if args.hyperopt or config.training.hyperopt:
            results = trainer.optimize_hyperparameters()
            logger.info(f"Hyperparameter optimization results: {results}")
        
        elif args.curriculum or config.training.curriculum_learning:
            results = trainer.train_with_curriculum()
            logger.info(f"Curriculum learning results: {results}")
        
        else:
            results = trainer.train_single_run()
            logger.info(f"Training results: {results}")
        
        # Save final results
        results_path = Path(config.training.save_path) / "final_results.yaml"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Training completed successfully. Results saved to {results_path}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()