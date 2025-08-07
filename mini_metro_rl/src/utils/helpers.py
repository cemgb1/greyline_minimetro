"""
Helper functions and utilities for Mini Metro RL project.

Provides common utility functions for data processing, mathematical operations,
visualization helpers, and other shared functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import json
import pickle
import time
from collections import defaultdict, deque
import psutil
import os

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Set random seeds to {seed}")


def normalize_array(arr: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize array using different methods.
    
    Args:
        arr: Array to normalize
        method: Normalization method ("minmax", "zscore", "robust")
        
    Returns:
        Normalized array
    """
    if method == "minmax":
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    elif method == "zscore":
        mean, std = arr.mean(), arr.std()
        if std == 0:
            return np.zeros_like(arr)
        return (arr - mean) / std
    
    elif method == "robust":
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            return np.zeros_like(arr)
        return (arr - median) / mad
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def moving_average(data: List[float], window_size: int) -> List[float]:
    """
    Calculate moving average of data.
    
    Args:
        data: List of values
        window_size: Size of moving window
        
    Returns:
        List of moving averages
    """
    if window_size <= 0 or window_size > len(data):
        return data.copy()
    
    result = []
    window = deque(maxlen=window_size)
    
    for value in data:
        window.append(value)
        result.append(sum(window) / len(window))
    
    return result


def exponential_decay(initial_value: float, decay_rate: float, step: int) -> float:
    """
    Calculate exponentially decayed value.
    
    Args:
        initial_value: Starting value
        decay_rate: Decay rate per step
        step: Current step
        
    Returns:
        Decayed value
    """
    return initial_value * (decay_rate ** step)


def linear_decay(
    initial_value: float, 
    final_value: float, 
    current_step: int, 
    total_steps: int
) -> float:
    """
    Calculate linearly decayed value.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        current_step: Current step
        total_steps: Total number of steps
        
    Returns:
        Linearly interpolated value
    """
    if current_step >= total_steps:
        return final_value
    
    progress = current_step / total_steps
    return initial_value + (final_value - initial_value) * progress


def calculate_euclidean_distance(
    pos1: Tuple[float, float], 
    pos2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def calculate_manhattan_distance(
    pos1: Tuple[float, float], 
    pos2: Tuple[float, float]
) -> float:
    """
    Calculate Manhattan distance between two points.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def create_network_graph(
    stations: Dict[int, Dict], 
    lines: Dict[int, Dict]
) -> nx.Graph:
    """
    Create NetworkX graph from station and line data.
    
    Args:
        stations: Dictionary of station data
        lines: Dictionary of line data
        
    Returns:
        NetworkX graph representation
    """
    G = nx.Graph()
    
    # Add stations as nodes
    for station_id, station_data in stations.items():
        position = station_data.get('position', (0, 0))
        station_type = station_data.get('type', 'unknown')
        
        G.add_node(
            station_id,
            pos=position,
            station_type=station_type,
            **station_data
        )
    
    # Add lines as edges
    for line_id, line_data in lines.items():
        line_stations = line_data.get('stations', [])
        line_color = line_data.get('color', 'black')
        
        # Connect consecutive stations
        for i in range(len(line_stations) - 1):
            station1 = line_stations[i]
            station2 = line_stations[i + 1]
            
            if station1 in G.nodes and station2 in G.nodes:
                # Calculate edge weight (distance)
                pos1 = G.nodes[station1]['pos']
                pos2 = G.nodes[station2]['pos']
                weight = calculate_euclidean_distance(pos1, pos2)
                
                G.add_edge(
                    station1, 
                    station2,
                    line_id=line_id,
                    color=line_color,
                    weight=weight
                )
        
        # For loops, connect last station to first
        if (line_data.get('line_type') == 'loop' and 
            len(line_stations) > 2):
            first_station = line_stations[0]
            last_station = line_stations[-1]
            
            if first_station in G.nodes and last_station in G.nodes:
                pos1 = G.nodes[first_station]['pos']
                pos2 = G.nodes[last_station]['pos']
                weight = calculate_euclidean_distance(pos1, pos2)
                
                G.add_edge(
                    first_station,
                    last_station,
                    line_id=line_id,
                    color=line_color,
                    weight=weight
                )
    
    return G


def calculate_network_metrics(graph: nx.Graph) -> Dict[str, float]:
    """
    Calculate network topology metrics.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Dictionary of network metrics
    """
    if len(graph.nodes) == 0:
        return {}
    
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = len(graph.nodes)
    metrics['num_edges'] = len(graph.edges)
    metrics['density'] = nx.density(graph)
    
    # Connectivity metrics
    if nx.is_connected(graph):
        metrics['is_connected'] = 1.0
        metrics['diameter'] = nx.diameter(graph)
        metrics['average_path_length'] = nx.average_shortest_path_length(graph)
        metrics['global_efficiency'] = nx.global_efficiency(graph)
    else:
        metrics['is_connected'] = 0.0
        metrics['diameter'] = float('inf')
        metrics['average_path_length'] = float('inf')
        metrics['global_efficiency'] = 0.0
    
    # Centrality metrics
    try:
        betweenness = nx.betweenness_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        degree = nx.degree_centrality(graph)
        
        metrics['avg_betweenness_centrality'] = np.mean(list(betweenness.values()))
        metrics['avg_closeness_centrality'] = np.mean(list(closeness.values()))
        metrics['avg_degree_centrality'] = np.mean(list(degree.values()))
        
        # Variance in centrality (indicates network balance)
        metrics['betweenness_variance'] = np.var(list(betweenness.values()))
        metrics['closeness_variance'] = np.var(list(closeness.values()))
        
    except Exception as e:
        logger.warning(f"Error calculating centrality metrics: {e}")
        for key in ['avg_betweenness_centrality', 'avg_closeness_centrality', 
                   'avg_degree_centrality', 'betweenness_variance', 'closeness_variance']:
            metrics[key] = 0.0
    
    return metrics


def save_object(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to file using pickle.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.debug(f"Saved object to {filepath}")


def load_object(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logger.debug(f"Loaded object from {filepath}")
    return obj


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON from {filepath}")
    return data


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """
    Format number with appropriate units (K, M, B).
    
    Args:
        number: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if abs(number) >= 1e9:
        return f"{number/1e9:.{precision}f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.{precision}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for monitoring.
    
    Returns:
        Dictionary with system information
    """
    info = {}
    
    # CPU information
    info['cpu_count'] = psutil.cpu_count()
    info['cpu_percent'] = psutil.cpu_percent(interval=1)
    
    # Memory information
    memory = psutil.virtual_memory()
    info['memory_total'] = memory.total
    info['memory_available'] = memory.available
    info['memory_percent'] = memory.percent
    
    # Disk information
    disk = psutil.disk_usage('/')
    info['disk_total'] = disk.total
    info['disk_free'] = disk.free
    info['disk_percent'] = (disk.used / disk.total) * 100
    
    # Process information
    process = psutil.Process(os.getpid())
    info['process_memory'] = process.memory_info().rss
    info['process_cpu_percent'] = process.cpu_percent()
    
    return info


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Timer", log_result: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Name for the timer
            log_result: Whether to log the result
        """
        self.name = name
        self.log_result = log_result
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and optionally log result."""
        self.end_time = time.time()
        if self.log_result:
            elapsed = self.elapsed_time()
            logger.info(f"{self.name} took {format_time(elapsed)}")
    
    def elapsed_time(self) -> float:
        """
        Get elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time


class PerformanceMonitor:
    """Monitor performance metrics over time."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of records to keep
        """
        self.max_history = max_history
        self.metrics = defaultdict(list)
        self.timestamps = []
    
    def record(self, **kwargs) -> None:
        """
        Record performance metrics.
        
        Args:
            **kwargs: Metric name-value pairs
        """
        timestamp = time.time()
        self.timestamps.append(timestamp)
        
        for name, value in kwargs.items():
            self.metrics[name].append(value)
        
        # Trim history if needed
        if len(self.timestamps) > self.max_history:
            excess = len(self.timestamps) - self.max_history
            self.timestamps = self.timestamps[excess:]
            
            for name in self.metrics:
                self.metrics[name] = self.metrics[name][excess:]
    
    def get_recent_average(self, metric_name: str, window: int = 100) -> float:
        """
        Get recent average of a metric.
        
        Args:
            metric_name: Name of the metric
            window: Window size for average
            
        Returns:
            Recent average value
        """
        if metric_name not in self.metrics:
            return 0.0
        
        values = self.metrics[metric_name]
        if not values:
            return 0.0
        
        recent_values = values[-window:]
        return np.mean(recent_values)
    
    def get_trend(self, metric_name: str, window: int = 100) -> str:
        """
        Get trend direction for a metric.
        
        Args:
            metric_name: Name of the metric
            window: Window size for trend analysis
            
        Returns:
            Trend direction ("up", "down", "stable")
        """
        if metric_name not in self.metrics:
            return "stable"
        
        values = self.metrics[metric_name]
        if len(values) < 2:
            return "stable"
        
        recent_values = values[-window:]
        if len(recent_values) < 2:
            return "stable"
        
        # Simple linear regression slope
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        threshold = np.std(recent_values) * 0.1  # 10% of standard deviation
        
        if slope > threshold:
            return "up"
        elif slope < -threshold:
            return "down"
        else:
            return "stable"
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'recent_avg': self.get_recent_average(name),
                    'trend': self.get_trend(name)
                }
        
        return summary


def create_colormap(n_colors: int, colormap_name: str = 'tab10') -> List[str]:
    """
    Create list of colors for visualization.
    
    Args:
        n_colors: Number of colors needed
        colormap_name: Name of matplotlib colormap
        
    Returns:
        List of hex color strings
    """
    try:
        cmap = plt.cm.get_cmap(colormap_name)
        colors = []
        
        for i in range(n_colors):
            color = cmap(i / max(1, n_colors - 1))
            hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
            colors.append(hex_color)
        
        return colors
    except Exception:
        # Fallback to simple colors
        basic_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', 
                       '#FF00FF', '#00FFFF', '#800000', '#008000']
        return (basic_colors * ((n_colors // len(basic_colors)) + 1))[:n_colors]


def interpolate_positions(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    progress: float
) -> Tuple[float, float]:
    """
    Interpolate between two positions.
    
    Args:
        start_pos: Starting position (x, y)
        end_pos: Ending position (x, y)
        progress: Progress from 0.0 to 1.0
        
    Returns:
        Interpolated position
    """
    progress = max(0.0, min(1.0, progress))
    
    x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
    y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
    
    return (x, y)