"""
TensorBoard logging system for Mini Metro RL.

Provides comprehensive logging of training metrics, network visualizations,
custom metrics, and performance analysis for TensorBoard monitoring.
"""

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from pathlib import Path
import io
from PIL import Image

from ..utils.helpers import create_network_graph, calculate_network_metrics

logger = logging.getLogger(__name__)


class TensorboardLogger:
    """
    Comprehensive TensorBoard logging for Mini Metro RL.
    
    Features:
    - Training metrics logging
    - Custom game-specific metrics
    - Network topology visualization
    - Q-value heatmaps
    - Performance analysis
    - Model interpretation
    """
    
    def __init__(
        self,
        log_dir: str,
        log_graph: bool = True,
        log_images: bool = True,
        log_histograms: bool = False,
        flush_secs: int = 30
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            log_graph: Whether to log model graphs
            log_images: Whether to log images
            log_histograms: Whether to log parameter histograms
            flush_secs: How often to flush logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_graph = log_graph
        self.log_images = log_images
        self.log_histograms = log_histograms
        
        # Initialize SummaryWriter
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs
        )
        
        # Metric tracking
        self.episode_metrics = {}
        self.step_metrics = {}
        
        # Visualization settings
        plt.style.use('seaborn-v0_8')
        
        logger.info(f"Initialized TensorBoard logger: {log_dir}")
    
    def log_scalar(
        self,
        tag: str,
        value: Union[float, int],
        step: int,
        walltime: Optional[float] = None
    ) -> None:
        """
        Log a scalar value.
        
        Args:
            tag: Tag for the metric
            value: Scalar value
            step: Global step
            walltime: Wall time (optional)
        """
        self.writer.add_scalar(tag, value, step, walltime)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int]],
        step: int,
        walltime: Optional[float] = None
    ) -> None:
        """
        Log multiple scalars under one main tag.
        
        Args:
            main_tag: Main tag for grouping
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Global step
            walltime: Wall time (optional)
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step, walltime)
    
    def log_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        step: int,
        bins: str = 'tensorflow'
    ) -> None:
        """
        Log histogram of values.
        
        Args:
            tag: Tag for the histogram
            values: Values to histogram
            step: Global step
            bins: Binning method
        """
        if self.log_histograms:
            self.writer.add_histogram(tag, values, step, bins)
    
    def log_image(
        self,
        tag: str,
        img_tensor: Union[torch.Tensor, np.ndarray],
        step: int,
        walltime: Optional[float] = None,
        dataformats: str = 'CHW'
    ) -> None:
        """
        Log an image.
        
        Args:
            tag: Tag for the image
            img_tensor: Image tensor
            step: Global step
            walltime: Wall time (optional)
            dataformats: Data format ('CHW', 'HWC', etc.)
        """
        if self.log_images:
            self.writer.add_image(tag, img_tensor, step, walltime, dataformats)
    
    def log_figure(
        self,
        tag: str,
        figure: plt.Figure,
        step: int,
        close: bool = True
    ) -> None:
        """
        Log a matplotlib figure.
        
        Args:
            tag: Tag for the figure
            figure: Matplotlib figure
            step: Global step
            close: Whether to close figure after logging
        """
        if self.log_images:
            self.writer.add_figure(tag, figure, step, close)
    
    def log_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor) -> None:
        """
        Log model graph.
        
        Args:
            model: PyTorch model
            input_to_model: Sample input tensor
        """
        if self.log_graph:
            self.writer.add_graph(model, input_to_model)
    
    def log_episode_metrics(
        self,
        episode: int,
        metrics: Dict[str, Any],
        prefix: str = "episode"
    ) -> None:
        """
        Log episode-level metrics.
        
        Args:
            episode: Episode number
            metrics: Dictionary of metrics
            prefix: Prefix for metric tags
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"{prefix}/{key}", value, episode)
            elif isinstance(value, dict):
                # Nested metrics
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        self.log_scalar(f"{prefix}/{key}/{subkey}", subvalue, episode)
    
    def log_training_metrics(
        self,
        step: int,
        loss: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log training metrics.
        
        Args:
            step: Training step
            loss: Training loss
            additional_metrics: Additional metrics to log
        """
        self.log_scalar("training/loss", loss, step)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.log_scalar(f"training/{key}", value, step)
    
    def log_reward_breakdown(
        self,
        step: int,
        reward_components: Dict[str, float]
    ) -> None:
        """
        Log detailed reward breakdown.
        
        Args:
            step: Step number
            reward_components: Dictionary of reward components
        """
        # Log individual components
        for component, value in reward_components.items():
            self.log_scalar(f"rewards/{component}", value, step)
        
        # Log total reward
        total_reward = sum(reward_components.values())
        self.log_scalar("rewards/total", total_reward, step)
        
        # Log as grouped scalars
        self.log_scalars("rewards_breakdown", reward_components, step)
    
    def log_game_state_metrics(
        self,
        step: int,
        game_state: Dict[str, Any]
    ) -> None:
        """
        Log game-specific metrics.
        
        Args:
            step: Step number
            game_state: Current game state
        """
        # Extract statistics
        stats = game_state.get('statistics', {})
        
        # Log basic metrics
        basic_metrics = {
            'passengers_delivered': stats.get('passengers_delivered', 0),
            'passengers_spawned': stats.get('passengers_spawned', 0),
            'delivery_rate': stats.get('delivery_rate', 0.0),
            'average_satisfaction': stats.get('average_satisfaction', 0.0),
            'game_duration': stats.get('game_duration', 0.0),
        }
        
        for key, value in basic_metrics.items():
            self.log_scalar(f"game/{key}", value, step)
        
        # Log resource utilization
        resources = game_state.get('resources', {})
        if resources:
            available = resources.get('available', {})
            used = resources.get('used', {})
            
            for resource_type in ['lines', 'trains', 'carriages']:
                if resource_type in available and resource_type in used:
                    total = available[resource_type]
                    utilized = used[resource_type]
                    utilization = utilized / max(1, total)
                    
                    self.log_scalar(f"resources/{resource_type}_utilization", utilization, step)
                    self.log_scalar(f"resources/{resource_type}_used", utilized, step)
                    self.log_scalar(f"resources/{resource_type}_available", total, step)
        
        # Log network metrics
        self._log_network_metrics(step, game_state)
    
    def _log_network_metrics(
        self,
        step: int,
        game_state: Dict[str, Any]
    ) -> None:
        """Log network topology metrics."""
        try:
            stations = game_state.get('stations', {})
            lines = game_state.get('lines', {})
            
            if not stations or not lines:
                return
            
            # Create network graph
            graph = create_network_graph(stations, lines)
            
            # Calculate metrics
            metrics = calculate_network_metrics(graph)
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    self.log_scalar(f"network/{key}", value, step)
            
        except Exception as e:
            logger.warning(f"Error logging network metrics: {e}")
    
    def log_agent_metrics(
        self,
        step: int,
        agent_metrics: Dict[str, Any],
        agent_prefix: str = "agent"
    ) -> None:
        """
        Log agent-specific metrics.
        
        Args:
            step: Step number
            agent_metrics: Agent metrics dictionary
            agent_prefix: Prefix for agent metrics
        """
        for key, value in agent_metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"{agent_prefix}/{key}", value, step)
            elif isinstance(value, dict):
                # Handle nested metrics (e.g., performance summary)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        self.log_scalar(f"{agent_prefix}/{key}/{subkey}", subvalue, step)
    
    def log_network_weights(
        self,
        model: torch.nn.Module,
        step: int,
        prefix: str = "weights"
    ) -> None:
        """
        Log network weights and gradients.
        
        Args:
            model: PyTorch model
            step: Step number
            prefix: Prefix for weight tags
        """
        if not self.log_histograms:
            return
        
        for name, param in model.named_parameters():
            # Log weights
            self.log_histogram(f"{prefix}/{name}", param, step)
            
            # Log gradients if available
            if param.grad is not None:
                self.log_histogram(f"{prefix}/{name}_grad", param.grad, step)
                
                # Log gradient norms
                grad_norm = torch.norm(param.grad).item()
                self.log_scalar(f"{prefix}/{name}_grad_norm", grad_norm, step)
    
    def log_q_value_heatmap(
        self,
        q_values: np.ndarray,
        step: int,
        action_names: Optional[List[str]] = None
    ) -> None:
        """
        Log Q-value heatmap.
        
        Args:
            q_values: Q-values array [batch_size, action_dim]
            step: Step number
            action_names: Names for actions (optional)
        """
        if not self.log_images:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create heatmap
            im = ax.imshow(q_values.T, cmap='RdYlBu', aspect='auto')
            
            # Labels
            ax.set_xlabel('Batch Index')
            ax.set_ylabel('Action')
            ax.set_title('Q-Value Heatmap')
            
            # Action names if provided
            if action_names and len(action_names) == q_values.shape[1]:
                ax.set_yticks(range(len(action_names)))
                ax.set_yticklabels(action_names)
            
            # Colorbar
            plt.colorbar(im, ax=ax)
            
            # Log figure
            self.log_figure("analysis/q_value_heatmap", fig, step)
            
        except Exception as e:
            logger.warning(f"Error creating Q-value heatmap: {e}")
    
    def log_network_topology_visualization(
        self,
        step: int,
        game_state: Dict[str, Any]
    ) -> None:
        """
        Log network topology visualization.
        
        Args:
            step: Step number
            game_state: Current game state
        """
        if not self.log_images:
            return
        
        try:
            stations = game_state.get('stations', {})
            lines = game_state.get('lines', {})
            
            if not stations or not lines:
                return
            
            # Create network graph
            graph = create_network_graph(stations, lines)
            
            if len(graph.nodes) == 0:
                return
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get positions
            pos = nx.get_node_attributes(graph, 'pos')
            if not pos:
                pos = nx.spring_layout(graph)
            
            # Draw network
            nx.draw_networkx_nodes(
                graph, pos, ax=ax,
                node_color='lightblue',
                node_size=300,
                alpha=0.8
            )
            
            nx.draw_networkx_edges(
                graph, pos, ax=ax,
                edge_color='gray',
                alpha=0.6,
                width=2
            )
            
            nx.draw_networkx_labels(
                graph, pos, ax=ax,
                font_size=8,
                font_weight='bold'
            )
            
            ax.set_title(f"Metro Network Topology (Step {step})")
            ax.axis('off')
            
            # Log figure
            self.log_figure("visualization/network_topology", fig, step)
            
        except Exception as e:
            logger.warning(f"Error creating network topology visualization: {e}")
    
    def log_passenger_flow_diagram(
        self,
        step: int,
        flow_data: Dict[str, Any]
    ) -> None:
        """
        Log passenger flow diagram.
        
        Args:
            step: Step number
            flow_data: Passenger flow data
        """
        if not self.log_images:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create flow visualization
            # This is a simplified example - real implementation would be more complex
            origins = flow_data.get('origins', [])
            destinations = flow_data.get('destinations', [])
            flows = flow_data.get('flows', [])
            
            if origins and destinations and flows:
                # Create flow matrix
                flow_matrix = np.zeros((len(origins), len(destinations)))
                for i, j, flow in flows:
                    if i < len(origins) and j < len(destinations):
                        flow_matrix[i, j] = flow
                
                # Plot heatmap
                im = ax.imshow(flow_matrix, cmap='YlOrRd', aspect='auto')
                
                ax.set_xlabel('Destination Station')
                ax.set_ylabel('Origin Station')
                ax.set_title('Passenger Flow Matrix')
                
                plt.colorbar(im, ax=ax, label='Passenger Count')
                
                # Log figure
                self.log_figure("visualization/passenger_flow", fig, step)
        
        except Exception as e:
            logger.warning(f"Error creating passenger flow diagram: {e}")
    
    def log_performance_comparison(
        self,
        step: int,
        metrics_history: Dict[str, List[float]],
        window_size: int = 100
    ) -> None:
        """
        Log performance comparison over time.
        
        Args:
            step: Step number
            metrics_history: History of metrics
            window_size: Window size for moving average
        """
        if not self.log_images:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            plot_metrics = ['delivery_rate', 'average_satisfaction', 'passengers_delivered', 'efficiency']
            
            for i, metric in enumerate(plot_metrics):
                if metric in metrics_history and len(metrics_history[metric]) > 1:
                    ax = axes[i]
                    values = metrics_history[metric]
                    
                    # Plot raw values
                    ax.plot(values, alpha=0.3, label='Raw')
                    
                    # Plot moving average if enough data
                    if len(values) >= window_size:
                        moving_avg = np.convolve(
                            values, 
                            np.ones(window_size)/window_size, 
                            mode='valid'
                        )
                        ax.plot(range(window_size-1, len(values)), moving_avg, label='Moving Avg')
                    
                    ax.set_title(metric.replace('_', ' ').title())
                    ax.set_xlabel('Episode')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.log_figure("analysis/performance_comparison", fig, step)
            
        except Exception as e:
            logger.warning(f"Error creating performance comparison: {e}")
    
    def log_custom_metric(
        self,
        tag: str,
        value: Union[float, int, np.ndarray, torch.Tensor],
        step: int,
        metric_type: str = "scalar"
    ) -> None:
        """
        Log custom metric with automatic type detection.
        
        Args:
            tag: Metric tag
            value: Metric value
            step: Step number
            metric_type: Type of metric ('scalar', 'histogram', 'image')
        """
        try:
            if metric_type == "scalar" and isinstance(value, (int, float)):
                self.log_scalar(tag, value, step)
            
            elif metric_type == "histogram" and isinstance(value, (np.ndarray, torch.Tensor)):
                self.log_histogram(tag, value, step)
            
            elif metric_type == "image" and isinstance(value, (np.ndarray, torch.Tensor)):
                self.log_image(tag, value, step)
            
            else:
                logger.warning(f"Unsupported metric type or value for {tag}")
                
        except Exception as e:
            logger.warning(f"Error logging custom metric {tag}: {e}")
    
    def flush(self) -> None:
        """Flush all pending logs."""
        self.writer.flush()
    
    def close(self) -> None:
        """Close the logger and cleanup resources."""
        self.writer.close()
        logger.info("TensorBoard logger closed")