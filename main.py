#!/usr/bin/env python3
"""
Main entry point for Mini Metro RL project.

Provides command-line interface for training, evaluation, and visualization
of reinforcement learning agents on the Mini Metro game.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mini_metro_rl.src.utils.config import load_config, Config
from mini_metro_rl.src.utils.helpers import set_random_seeds, Timer
from mini_metro_rl.src.environment.metro_env import MetroEnvironment, EnvironmentConfig
from mini_metro_rl.src.agents.dqn_agent import DQNAgent
from mini_metro_rl.src.agents.ppo_agent import PPOAgent
from mini_metro_rl.src.agents.multi_agent import MultiAgent
from mini_metro_rl.src.visualization.pygame_renderer import PygameRenderer, RenderConfig
from mini_metro_rl.src.visualization.tensorboard_logger import TensorboardLogger

logger = logging.getLogger(__name__)


def setup_logging(config: Config) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Project configuration
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level.upper()),
        format=config.logging.log_format,
        filename=config.logging.log_file,
        filemode='a' if config.logging.log_file else None
    )
    
    # Also log to console if logging to file
    if config.logging.log_file:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.logging.log_level.upper()))
        console_formatter = logging.Formatter(config.logging.log_format)
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)


def create_environment(config: Config) -> MetroEnvironment:
    """
    Create the Mini Metro environment.
    
    Args:
        config: Project configuration
        
    Returns:
        Configured environment
    """
    # Convert config to environment config
    env_config = EnvironmentConfig(
        map_name=config.game.map_name,
        difficulty=config.game.difficulty,
        max_episode_steps=config.game.max_episode_steps,
        observation_type=config.environment.observation_type,
        action_space_type=config.environment.action_space_type,
        reward_weights=config.environment.reward_weights,
        real_time=config.game.real_time,
        time_step=config.game.time_step,
        seed=config.game.seed
    )
    
    env = MetroEnvironment(env_config)
    logger.info(f"Created environment: {config.game.map_name} ({config.game.difficulty})")
    
    return env


def create_agent(config: Config, env: MetroEnvironment, agent_type: str = None) -> Any:
    """
    Create the RL agent.
    
    Args:
        config: Project configuration
        env: Environment instance
        agent_type: Type of agent to create (override config)
        
    Returns:
        Configured agent
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.nvec[0]  # First dimension for action type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent_type = agent_type or config.multi_agent.agent_type
    
    if agent_type == "dqn":
        agent = DQNAgent(state_dim, action_dim, config.dqn, device)
        logger.info(f"Created DQN agent: state_dim={state_dim}, action_dim={action_dim}")
        
    elif agent_type == "ppo":
        agent = PPOAgent(state_dim, action_dim, config.ppo, device)
        logger.info(f"Created PPO agent: state_dim={state_dim}, action_dim={action_dim}")
        
    elif agent_type == "multi":
        agent_configs = {"dqn": config.dqn, "ppo": config.ppo}
        agent = MultiAgent(state_dim, action_dim, config.multi_agent, agent_configs, device)
        logger.info(f"Created Multi-agent system with {config.multi_agent.num_agents} agents")
        
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent


def train_agent(
    agent: Any,
    env: MetroEnvironment,
    config: Config,
    visualize: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train the RL agent.
    
    Args:
        agent: RL agent to train
        env: Environment
        config: Configuration
        visualize: Whether to show visualization
        save_path: Path to save trained model
        
    Returns:
        Training metrics
    """
    # Setup logging
    tb_logger = None
    if config.training.tensorboard_log:
        tb_logger = TensorboardLogger(
            config.logging.tensorboard_dir,
            config.logging.log_graph,
            config.logging.log_images,
            config.logging.log_histograms
        )
    
    # Setup visualization
    renderer = None
    if visualize:
        render_config = RenderConfig(
            width=config.visualization.resolution[0],
            height=config.visualization.resolution[1],
            fps=config.visualization.fps,
            show_passenger_destinations=config.visualization.show_passenger_destinations,
            show_train_loads=config.visualization.show_train_loads,
            show_station_queues=config.visualization.show_station_queues,
            show_performance_metrics=config.visualization.show_performance_metrics
        )
        renderer = PygameRenderer(render_config)
    
    # Training loop
    total_steps = 0
    episode = 0
    best_reward = float('-inf')
    
    logger.info(f"Starting training for {config.training.total_timesteps} steps")
    
    with Timer("Training", log_result=True):
        while total_steps < config.training.total_timesteps:
            episode += 1
            episode_reward = 0
            episode_steps = 0
            
            # Reset environment
            obs, info = env.reset()
            done = False
            
            while not done and episode_steps < config.game.max_episode_steps:
                # Get action
                if hasattr(agent, 'act'):
                    # Single agent
                    action_mask = env.get_action_mask()
                    if isinstance(agent, PPOAgent):
                        action, log_prob, value = agent.act(obs, action_mask)
                        action_info = {'action': action, 'log_prob': log_prob, 'value': value}
                    else:
                        action = agent.act(obs, action_mask)
                        action_info = {'action': action}
                else:
                    # Multi-agent
                    action_mask = env.get_action_mask()
                    actions = agent.act({0: obs}, {0: action_mask} if action_mask is not None else None)
                    action_info = actions.get(0, {'action': 0})
                    action = action_info['action']
                
                # Take step
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Update agent
                if hasattr(agent, 'step'):
                    if isinstance(agent, PPOAgent):
                        agent.step(
                            obs, action, reward, next_obs, done,
                            action_info.get('log_prob', 0.0),
                            action_info.get('value', 0.0)
                        )
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
                if tb_logger and total_steps % config.training.log_frequency == 0:
                    tb_logger.log_scalar("training/episode_reward", episode_reward, total_steps)
                    tb_logger.log_scalar("training/episode_steps", episode_steps, total_steps)
                    
                    # Log reward breakdown if available
                    if 'reward_breakdown' in step_info:
                        tb_logger.log_reward_breakdown(total_steps, step_info['reward_breakdown'])
                    
                    # Log game state metrics
                    game_state = env.game.get_detailed_state()
                    tb_logger.log_game_state_metrics(total_steps, game_state)
                
                # Visualization
                if renderer:
                    game_state = env.game.get_detailed_state()
                    renderer.render(game_state)
                    
                    # Handle events
                    events = renderer.handle_events()
                    if events['quit']:
                        logger.info("Visualization quit requested")
                        done = True
                        break
            
            # End episode
            if hasattr(agent, 'end_episode'):
                agent.end_episode()
            
            # Logging
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: reward={episode_reward:.2f}, steps={episode_steps}")
            
            # Save model
            if save_path and episode_reward > best_reward:
                best_reward = episode_reward
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                agent.save(save_path)
                logger.info(f"Saved best model with reward {best_reward:.2f}")
            
            # Periodic saving
            if save_path and episode % (config.training.save_frequency // 1000) == 0:
                checkpoint_path = f"{save_path}_episode_{episode}.pt"
                agent.save(checkpoint_path)
    
    # Cleanup
    if tb_logger:
        tb_logger.close()
    if renderer:
        renderer.close()
    
    return {
        'total_episodes': episode,
        'total_steps': total_steps,
        'best_reward': best_reward
    }


def evaluate_agent(
    agent: Any,
    env: MetroEnvironment,
    config: Config,
    num_episodes: int = 10,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    Evaluate the trained agent.
    
    Args:
        agent: Trained agent
        env: Environment
        config: Configuration
        num_episodes: Number of evaluation episodes
        visualize: Whether to show visualization
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating agent for {num_episodes} episodes")
    
    # Setup visualization
    renderer = None
    if visualize:
        render_config = RenderConfig(
            width=config.visualization.resolution[0],
            height=config.visualization.resolution[1],
            fps=config.visualization.fps,
            show_performance_metrics=True
        )
        renderer = PygameRenderer(render_config)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Get action (deterministic for evaluation)
            if hasattr(agent, 'act'):
                action_mask = env.get_action_mask()
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.act(obs, action_mask)
                else:
                    action = agent.act(obs, action_mask)
            else:
                # Multi-agent
                action_mask = env.get_action_mask()
                actions = agent.act({0: obs}, {0: action_mask} if action_mask is not None else None)
                action = actions.get(0, {'action': 0})['action']
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            # Visualization
            if renderer:
                game_state = env.game.get_detailed_state()
                renderer.render(game_state)
                
                events = renderer.handle_events()
                if events['quit']:
                    break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        logger.info(f"Eval Episode {episode + 1}: reward={episode_reward:.2f}, steps={episode_steps}")
    
    # Cleanup
    if renderer:
        renderer.close()
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    logger.info(f"Evaluation results: {metrics}")
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mini Metro RL")
    
    # Mode selection
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the agent")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    
    # Configuration
    parser.add_argument("--config", type=str, default="dqn_config", 
                       help="Configuration file name (without .yaml)")
    parser.add_argument("--config-dir", type=str, default=None,
                       help="Configuration directory")
    
    # Agent options
    parser.add_argument("--agent", type=str, choices=["dqn", "ppo", "multi"],
                       help="Agent type (overrides config)")
    parser.add_argument("--load-model", type=str, help="Path to load trained model")
    parser.add_argument("--save-model", type=str, help="Path to save trained model")
    
    # Training options
    parser.add_argument("--episodes", type=int, help="Number of episodes")
    parser.add_argument("--steps", type=int, help="Number of training steps")
    
    # Evaluation options
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    
    # Logging options
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Other options
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load configuration
    config_overrides = {}
    if args.log_level:
        config_overrides['logging'] = {'log_level': args.log_level}
    if args.seed:
        config_overrides['game'] = {'seed': args.seed}
    if args.episodes:
        config_overrides['training'] = {'total_episodes': args.episodes}
    if args.steps:
        config_overrides['training'] = {'total_timesteps': args.steps}
    
    config = load_config(args.config, args.config_dir, config_overrides)
    
    # Setup logging
    setup_logging(config)
    
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
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config.experiment_name}")
    
    try:
        # Create environment
        env = create_environment(config)
        
        # Create agent
        agent = create_agent(config, env, args.agent)
        
        # Load model if specified
        if args.load_model:
            agent.load(args.load_model)
            logger.info(f"Loaded model from {args.load_model}")
        
        # Run modes
        if args.train:
            save_path = args.save_model or f"./models/{config.experiment_name}.pt"
            train_metrics = train_agent(agent, env, config, args.visualize, save_path)
            logger.info(f"Training completed: {train_metrics}")
        
        if args.evaluate:
            eval_metrics = evaluate_agent(agent, env, config, args.eval_episodes, args.visualize)
            logger.info(f"Evaluation completed: {eval_metrics}")
        
        # Default: just visualize
        if not args.train and not args.evaluate and args.visualize:
            logger.info("Running visualization mode")
            evaluate_agent(agent, env, config, 1, True)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Program completed successfully")


if __name__ == "__main__":
    main()