#!/usr/bin/env python3
"""
Evaluation script for Mini Metro RL agents.

Comprehensive evaluation with performance analysis, visualization,
model comparison, and detailed reporting.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml

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

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis.
    
    Features:
    - Performance benchmarking
    - Statistical analysis
    - Visualization generation
    - Model comparison
    - Report generation
    """
    
    def __init__(self, config: Config, output_dir: str = "./evaluation_results"):
        """
        Initialize model evaluator.
        
        Args:
            config: Evaluation configuration
            output_dir: Directory for evaluation outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.evaluation_results = {}
        self.detailed_episodes = []
        
        logger.info(f"Initialized model evaluator, output dir: {output_dir}")
    
    def evaluate_model(
        self,
        model_path: str,
        agent_type: str,
        num_episodes: int = 100,
        visualize: bool = False,
        save_episodes: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model_path: Path to saved model
            agent_type: Type of agent
            num_episodes: Number of evaluation episodes
            visualize: Whether to show visualization
            save_episodes: Whether to save episode details
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model: {model_path}")
        
        # Create environment
        env = self._create_environment()
        
        # Create and load agent
        agent = self._create_agent(env, agent_type)
        agent.load(model_path)
        
        # Setup visualization
        renderer = None
        if visualize:
            render_config = RenderConfig(
                width=self.config.visualization.resolution[0],
                height=self.config.visualization.resolution[1],
                fps=self.config.visualization.fps,
                show_performance_metrics=True
            )
            renderer = PygameRenderer(render_config)
        
        # Run evaluation episodes
        episode_results = []
        
        with Timer(f"Evaluating {num_episodes} episodes", log_result=True):
            for episode in range(num_episodes):
                episode_result = self._run_single_episode(
                    agent, env, episode, renderer, save_episodes
                )
                episode_results.append(episode_result)
                
                if episode % 10 == 0:
                    logger.info(f"Completed episode {episode + 1}/{num_episodes}")
        
        # Cleanup
        if renderer:
            renderer.close()
        
        # Calculate statistics
        results = self._calculate_statistics(episode_results)
        results['model_path'] = model_path
        results['agent_type'] = agent_type
        results['num_episodes'] = num_episodes
        
        # Store results
        model_name = Path(model_path).stem
        self.evaluation_results[model_name] = results
        
        logger.info(f"Evaluation completed: {results['summary']}")
        return results
    
    def _create_environment(self) -> MetroEnvironment:
        """Create evaluation environment."""
        env_config = EnvironmentConfig(
            map_name=self.config.game.map_name,
            difficulty=self.config.game.difficulty,
            max_episode_steps=self.config.game.max_episode_steps,
            observation_type=self.config.environment.observation_type,
            action_space_type=self.config.environment.action_space_type,
            reward_weights=self.config.environment.reward_weights,
            real_time=False,  # Always false for evaluation
            time_step=self.config.game.time_step,
            seed=self.config.game.seed
        )
        
        return MetroEnvironment(env_config)
    
    def _create_agent(self, env: MetroEnvironment, agent_type: str) -> Any:
        """Create agent for evaluation."""
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.nvec[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if agent_type == "dqn":
            return DQNAgent(state_dim, action_dim, self.config.dqn, device)
        elif agent_type == "ppo":
            return PPOAgent(state_dim, action_dim, self.config.ppo, device)
        elif agent_type == "multi":
            agent_configs = {"dqn": self.config.dqn, "ppo": self.config.ppo}
            return MultiAgent(state_dim, action_dim, self.config.multi_agent, agent_configs, device)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _run_single_episode(
        self,
        agent: Any,
        env: MetroEnvironment,
        episode_id: int,
        renderer: Optional[PygameRenderer] = None,
        save_details: bool = False
    ) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Episode tracking
        episode_data = {
            'episode_id': episode_id,
            'steps': [],
            'actions': [],
            'rewards': [],
            'game_states': []
        }
        
        while not done and episode_steps < self.config.game.max_episode_steps:
            # Get action (deterministic for evaluation)
            action_mask = env.get_action_mask()
            
            if hasattr(agent, 'act'):
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.act(obs, action_mask)
                else:
                    # For DQN, temporarily disable exploration
                    if hasattr(agent, 'epsilon'):
                        old_epsilon = agent.epsilon
                        agent.epsilon = 0.0
                    
                    action = agent.act(obs, action_mask)
                    
                    if hasattr(agent, 'epsilon'):
                        agent.epsilon = old_epsilon
            else:
                # Multi-agent
                actions = agent.act({0: obs}, {0: action_mask} if action_mask is not None else None)
                action = actions.get(0, {'action': 0})['action']
            
            # Take step
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Track episode data
            if save_details:
                episode_data['steps'].append(episode_steps)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['game_states'].append(env.game.get_detailed_state())
            
            # Update tracking
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            # Visualization
            if renderer:
                game_state = env.game.get_detailed_state()
                renderer.render(game_state)
                
                events = renderer.handle_events()
                if events['quit']:
                    break
        
        # Calculate episode statistics
        final_game_state = env.game.get_detailed_state()
        stats = final_game_state.get('statistics', {})
        
        episode_result = {
            'episode_id': episode_id,
            'total_reward': episode_reward,
            'episode_steps': episode_steps,
            'passengers_delivered': stats.get('passengers_delivered', 0),
            'passengers_spawned': stats.get('passengers_spawned', 0),
            'delivery_rate': stats.get('delivery_rate', 0.0),
            'average_satisfaction': stats.get('average_satisfaction', 0.0),
            'game_duration': stats.get('game_duration', 0.0),
            'current_week': final_game_state.get('current_week', 1),
            'stations_count': len(final_game_state.get('stations', {})),
            'lines_count': len(final_game_state.get('lines', {})),
            'game_over': final_game_state.get('game_state') == 'game_over'
        }
        
        # Store detailed episode data if requested
        if save_details:
            episode_data.update(episode_result)
            self.detailed_episodes.append(episode_data)
        
        return episode_result
    
    def _calculate_statistics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics from episode results."""
        df = pd.DataFrame(episode_results)
        
        # Basic statistics
        stats = {
            'summary': {
                'mean_reward': df['total_reward'].mean(),
                'std_reward': df['total_reward'].std(),
                'min_reward': df['total_reward'].min(),
                'max_reward': df['total_reward'].max(),
                'median_reward': df['total_reward'].median(),
                'mean_steps': df['episode_steps'].mean(),
                'std_steps': df['episode_steps'].std(),
                'success_rate': (df['game_over'] == False).mean(),
                'mean_delivery_rate': df['delivery_rate'].mean(),
                'mean_satisfaction': df['average_satisfaction'].mean(),
                'mean_final_week': df['current_week'].mean(),
                'mean_stations': df['stations_count'].mean(),
                'mean_lines': df['lines_count'].mean()
            },
            'percentiles': {
                'reward_25': df['total_reward'].quantile(0.25),
                'reward_50': df['total_reward'].quantile(0.50),
                'reward_75': df['total_reward'].quantile(0.75),
                'reward_95': df['total_reward'].quantile(0.95),
                'steps_25': df['episode_steps'].quantile(0.25),
                'steps_50': df['episode_steps'].quantile(0.50),
                'steps_75': df['episode_steps'].quantile(0.75),
                'steps_95': df['episode_steps'].quantile(0.95)
            },
            'correlations': {
                'reward_delivery_rate': df['total_reward'].corr(df['delivery_rate']),
                'reward_satisfaction': df['total_reward'].corr(df['average_satisfaction']),
                'reward_steps': df['total_reward'].corr(df['episode_steps']),
                'delivery_satisfaction': df['delivery_rate'].corr(df['average_satisfaction'])
            },
            'raw_data': episode_results
        }
        
        return stats
    
    def compare_models(self, model_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple model evaluations.
        
        Args:
            model_evaluations: Dictionary of model name -> evaluation results
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(model_evaluations)} models")
        
        # Extract comparison data
        comparison_data = []
        for model_name, results in model_evaluations.items():
            summary = results['summary']
            comparison_data.append({
                'model': model_name,
                'agent_type': results.get('agent_type', 'unknown'),
                **summary
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Rankings
        ranking_metrics = ['mean_reward', 'success_rate', 'mean_delivery_rate', 'mean_satisfaction']
        rankings = {}
        
        for metric in ranking_metrics:
            if metric in df.columns:
                rankings[metric] = df.nlargest(len(df), metric)[['model', metric]].to_dict('records')
        
        # Statistical comparisons
        comparisons = {
            'rankings': rankings,
            'summary_table': df.to_dict('records'),
            'best_overall': df.loc[df['mean_reward'].idxmax()]['model'],
            'most_consistent': df.loc[df['std_reward'].idxmin()]['model'],
            'highest_success_rate': df.loc[df['success_rate'].idxmax()]['model']
        }
        
        return comparisons
    
    def generate_visualizations(self, model_name: str, results: Dict[str, Any]) -> List[str]:
        """
        Generate visualization plots for evaluation results.
        
        Args:
            model_name: Name of the model
            results: Evaluation results
            
        Returns:
            List of generated plot file paths
        """
        plot_paths = []
        episode_data = pd.DataFrame(results['raw_data'])
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Reward distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(episode_data['total_reward'], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(results['summary']['mean_reward'], color='red', linestyle='--', 
                  label=f"Mean: {results['summary']['mean_reward']:.2f}")
        ax.axvline(results['summary']['median_reward'], color='orange', linestyle='--',
                  label=f"Median: {results['summary']['median_reward']:.2f}")
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Reward Distribution - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / f"{model_name}_reward_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        # 2. Performance over episodes
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward over episodes
        axes[0, 0].plot(episode_data['total_reward'], alpha=0.7)
        axes[0, 0].plot(episode_data['total_reward'].rolling(10).mean(), color='red', linewidth=2)
        axes[0, 0].set_title('Reward Over Episodes')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Delivery rate over episodes
        axes[0, 1].plot(episode_data['delivery_rate'], alpha=0.7, color='green')
        axes[0, 1].plot(episode_data['delivery_rate'].rolling(10).mean(), color='darkgreen', linewidth=2)
        axes[0, 1].set_title('Delivery Rate Over Episodes')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Delivery Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Satisfaction over episodes
        axes[1, 0].plot(episode_data['average_satisfaction'], alpha=0.7, color='purple')
        axes[1, 0].plot(episode_data['average_satisfaction'].rolling(10).mean(), color='darkpurple', linewidth=2)
        axes[1, 0].set_title('Satisfaction Over Episodes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Satisfaction')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Episode length over episodes
        axes[1, 1].plot(episode_data['episode_steps'], alpha=0.7, color='orange')
        axes[1, 1].plot(episode_data['episode_steps'].rolling(10).mean(), color='darkorange', linewidth=2)
        axes[1, 1].set_title('Episode Length Over Episodes')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{model_name}_performance_over_time.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        # 3. Correlation heatmap
        correlation_data = episode_data[['total_reward', 'delivery_rate', 'average_satisfaction', 
                                       'episode_steps', 'passengers_delivered', 'current_week']].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f'Performance Metrics Correlation - {model_name}')
        
        plot_path = self.output_dir / f"{model_name}_correlation_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        return plot_paths
    
    def generate_report(self, model_evaluations: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_evaluations: Dictionary of model evaluations
            
        Returns:
            Path to generated report
        """
        report_path = self.output_dir / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Mini Metro RL Model Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Executive Summary\n\n")
            f.write(f"Evaluated {len(model_evaluations)} models on Mini Metro RL task.\n\n")
            
            # Model comparison
            if len(model_evaluations) > 1:
                comparison = self.compare_models(model_evaluations)
                f.write("## Model Comparison\n\n")
                f.write("### Overall Rankings\n\n")
                
                for metric, ranking in comparison['rankings'].items():
                    f.write(f"**{metric.replace('_', ' ').title()}:**\n")
                    for i, entry in enumerate(ranking[:5]):  # Top 5
                        f.write(f"{i+1}. {entry['model']}: {entry[metric]:.4f}\n")
                    f.write("\n")
                
                f.write(f"- **Best Overall Model:** {comparison['best_overall']}\n")
                f.write(f"- **Most Consistent Model:** {comparison['most_consistent']}\n")
                f.write(f"- **Highest Success Rate:** {comparison['highest_success_rate']}\n\n")
            
            # Individual model results
            f.write("## Individual Model Results\n\n")
            
            for model_name, results in model_evaluations.items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- **Agent Type:** {results.get('agent_type', 'Unknown')}\n")
                f.write(f"- **Episodes Evaluated:** {results['num_episodes']}\n\n")
                
                summary = results['summary']
                f.write("**Performance Metrics:**\n")
                f.write(f"- Mean Reward: {summary['mean_reward']:.4f} ± {summary['std_reward']:.4f}\n")
                f.write(f"- Success Rate: {summary['success_rate']:.2%}\n")
                f.write(f"- Delivery Rate: {summary['mean_delivery_rate']:.2%}\n")
                f.write(f"- Passenger Satisfaction: {summary['mean_satisfaction']:.4f}\n")
                f.write(f"- Average Episode Length: {summary['mean_steps']:.1f} steps\n")
                f.write(f"- Average Final Week: {summary['mean_final_week']:.1f}\n\n")
                
                # Generate and link visualizations
                plot_paths = self.generate_visualizations(model_name, results)
                if plot_paths:
                    f.write("**Visualizations:**\n")
                    for plot_path in plot_paths:
                        plot_name = Path(plot_path).name
                        f.write(f"- ![{plot_name}]({plot_name})\n")
                    f.write("\n")
        
        logger.info(f"Generated evaluation report: {report_path}")
        return str(report_path)
    
    def save_results(self, filename: str = "evaluation_results.yaml") -> str:
        """
        Save evaluation results to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            yaml.dump(self.evaluation_results, f, default_flow_style=False)
        
        logger.info(f"Saved evaluation results to {output_path}")
        return str(output_path)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Mini Metro RL Model Evaluation")
    
    # Configuration
    parser.add_argument("--config", type=str, default="dqn_config",
                       help="Configuration file name")
    parser.add_argument("--config-dir", type=str, default=None,
                       help="Configuration directory")
    
    # Model options
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model file or directory with multiple models")
    parser.add_argument("--agent-type", type=str, choices=["dqn", "ppo", "multi"],
                       help="Agent type (auto-detect if not specified)")
    
    # Evaluation options
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization during evaluation")
    parser.add_argument("--save-episodes", action="store_true",
                       help="Save detailed episode data")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate comprehensive evaluation report")
    
    # Other options
    parser.add_argument("--seed", type=int, help="Random seed for evaluation")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config_overrides = {}
    if args.seed:
        config_overrides['game'] = {'seed': args.seed}
    
    config = load_config(args.config, args.config_dir, config_overrides)
    
    # Set random seeds
    if config.game.seed:
        set_random_seeds(config.game.seed)
    
    logger.info(f"Starting evaluation with config: {config.experiment_name}")
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(config, args.output_dir)
        
        # Determine models to evaluate
        model_path = Path(args.model)
        
        if model_path.is_file():
            # Single model
            agent_type = args.agent_type
            if not agent_type:
                # Try to auto-detect from filename
                if 'dqn' in model_path.name.lower():
                    agent_type = 'dqn'
                elif 'ppo' in model_path.name.lower():
                    agent_type = 'ppo'
                else:
                    agent_type = 'dqn'  # Default
            
            results = evaluator.evaluate_model(
                str(model_path),
                agent_type,
                args.episodes,
                args.visualize,
                args.save_episodes
            )
            
            logger.info(f"Evaluation completed: {results['summary']}")
        
        elif model_path.is_dir():
            # Multiple models in directory
            model_files = list(model_path.glob("*.pt"))
            
            if not model_files:
                logger.error(f"No .pt files found in {model_path}")
                sys.exit(1)
            
            logger.info(f"Found {len(model_files)} models to evaluate")
            
            for model_file in model_files:
                # Auto-detect agent type
                agent_type = args.agent_type
                if not agent_type:
                    if 'dqn' in model_file.name.lower():
                        agent_type = 'dqn'
                    elif 'ppo' in model_file.name.lower():
                        agent_type = 'ppo'
                    else:
                        agent_type = 'dqn'
                
                logger.info(f"Evaluating {model_file.name} as {agent_type} agent")
                
                results = evaluator.evaluate_model(
                    str(model_file),
                    agent_type,
                    args.episodes,
                    args.visualize and len(model_files) == 1,  # Only visualize if single model
                    args.save_episodes
                )
                
                logger.info(f"Completed {model_file.name}: {results['summary']['mean_reward']:.4f}")
        
        else:
            logger.error(f"Model path not found: {model_path}")
            sys.exit(1)
        
        # Save results
        evaluator.save_results()
        
        # Generate report if requested
        if args.generate_report:
            report_path = evaluator.generate_report(evaluator.evaluation_results)
            logger.info(f"Generated evaluation report: {report_path}")
        
        logger.info("Evaluation completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()