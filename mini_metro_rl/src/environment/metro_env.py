"""
Mini Metro Gymnasium Environment for Reinforcement Learning.

Implements a complete RL environment with comprehensive state representation,
action space, and reward system for training RL agents on Mini Metro.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from dataclasses import dataclass
from enum import Enum

from ..game.mini_metro_game import MiniMetroGame, GameDifficulty, GameState
from ..game.station import StationType
from ..game.line import LineColor
from .rewards import RewardCalculator, RewardWeights

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Available action types in the environment."""
    NO_ACTION = 0
    CREATE_LINE = 1
    EXTEND_LINE = 2
    REMOVE_STATION_FROM_LINE = 3
    ADD_TRAIN = 4
    REMOVE_TRAIN = 5
    ADD_CARRIAGE = 6
    CONVERT_TO_LOOP = 7
    CONVERT_TO_LINEAR = 8
    ADD_BRIDGE_TUNNEL = 9
    REORDER_LINE_STATIONS = 10


@dataclass
class EnvironmentConfig:
    """Configuration for the Metro environment."""
    map_name: str = "london"
    difficulty: GameDifficulty = GameDifficulty.NORMAL
    max_episode_steps: int = 3000
    observation_type: str = "vector"  # "vector", "image", or "graph"
    action_space_type: str = "discrete"  # "discrete" or "continuous"
    reward_weights: Optional[RewardWeights] = None
    real_time: bool = False
    time_step: float = 1.0  # Seconds per step
    seed: Optional[int] = None


class MetroEnvironment(gym.Env):
    """
    Gymnasium environment for Mini Metro reinforcement learning.
    
    Features:
    - Comprehensive state representation (stations, lines, trains, passengers)
    - Complete action space covering all game mechanics
    - Multi-objective reward system
    - Support for different observation types
    - Configurable difficulty and game settings
    - Proper episode management and termination
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'ansi'],
        'render_fps': 30
    }
    
    def __init__(self, config: EnvironmentConfig = None):
        """
        Initialize the Metro environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config or EnvironmentConfig()
        
        # Initialize game
        self.game = MiniMetroGame(
            map_name=self.config.map_name,
            difficulty=self.config.difficulty,
            real_time=self.config.real_time,
            seed=self.config.seed
        )
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(self.config.reward_weights)
        
        # Episode management
        self.episode_step = 0
        self.max_episode_steps = self.config.max_episode_steps
        self.episode_rewards = []
        self.episode_info = {}
        
        # Define action and observation spaces
        self._define_action_space()
        self._define_observation_space()
        
        # Rendering
        self.render_mode = None
        self.renderer = None
        
        logger.info(f"Initialized Metro environment with config: {self.config}")
    
    def _define_action_space(self) -> None:
        """Define the action space based on configuration."""
        if self.config.action_space_type == "discrete":
            # Discrete action space with action type and parameters
            # Action format: [action_type, param1, param2, param3]
            # param1-3 are context-dependent (station IDs, line IDs, etc.)
            
            max_stations = 20  # Maximum expected stations
            max_lines = 10     # Maximum lines
            max_trains = 20    # Maximum trains
            
            self.action_space = spaces.MultiDiscrete([
                len(ActionType),     # Action type
                max_stations,        # Parameter 1 (usually station/line ID)
                max_stations,        # Parameter 2 (usually station ID)
                max_lines,           # Parameter 3 (usually line ID or position)
            ])
        else:
            raise NotImplementedError("Continuous action space not yet implemented")
    
    def _define_observation_space(self) -> None:
        """Define the observation space based on configuration."""
        if self.config.observation_type == "vector":
            # Vector observation with comprehensive game state
            # This will be dynamically sized based on game state
            # For now, define a reasonable upper bound
            
            max_stations = 20
            max_lines = 10
            max_trains = 20
            
            # Game state vector components:
            # - Global state (time, week, resources, performance): ~15 values
            # - Station states (per station): ~15 values each
            # - Line states (per line): ~10 values each  
            # - Train states (per train): ~12 values each
            
            global_state_size = 15
            station_state_size = max_stations * 15
            line_state_size = max_lines * 10
            train_state_size = max_trains * 12
            
            total_size = (global_state_size + station_state_size + 
                         line_state_size + train_state_size)
            
            self.observation_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(total_size,),
                dtype=np.float32
            )
            
        elif self.config.observation_type == "image":
            # Image observation (rendered game state)
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(600, 800, 3),  # Height, Width, Channels
                dtype=np.uint8
            )
            
        else:
            raise NotImplementedError(f"Observation type {self.config.observation_type} not implemented")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for the episode
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.config.seed = seed
        
        # Reset game
        self.game.reset()
        self.game.start_game()
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_info = {}
        
        # Reset reward calculator
        self.reward_calculator.reset(self.game.current_time)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug("Environment reset")
        return observation, info
    
    def step(
        self, 
        action: Union[np.ndarray, int, List[int]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Store previous state for reward calculation
        previous_state = self.game.get_detailed_state()
        
        # Execute action
        action_info = self._execute_action(action)
        
        # Update game state
        self.game.update(self.config.time_step)
        
        # Get new state
        current_state = self.game.get_detailed_state()
        
        # Calculate reward
        reward, reward_breakdown = self.reward_calculator.calculate_reward(
            current_state, action_info, self.game.is_game_over()
        )
        
        # Check termination conditions
        terminated = self.game.is_game_over()
        truncated = self.episode_step >= self.max_episode_steps
        
        # Update episode tracking
        self.episode_step += 1
        self.episode_rewards.append(reward)
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'action_info': action_info,
            'reward_breakdown': reward_breakdown,
            'game_state': current_state['game_state']
        })
        
        # Log step information
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Step {self.episode_step}: action={action_info['type']}, "
                        f"reward={reward:.3f}, terminated={terminated}")
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: Union[np.ndarray, int, List[int]]) -> Dict[str, Any]:
        """
        Execute the given action in the game.
        
        Args:
            action: Action to execute
            
        Returns:
            Dictionary with action information and success status
        """
        # Parse action
        if isinstance(action, int):
            action = [action, 0, 0, 0]
        elif isinstance(action, np.ndarray):
            action = action.tolist()
        
        action_type = ActionType(action[0])
        param1, param2, param3 = action[1], action[2], action[3]
        
        action_info = {
            'type': action_type.name.lower(),
            'params': [param1, param2, param3],
            'success': False,
            'message': ''
        }
        
        try:
            # Execute action based on type
            if action_type == ActionType.NO_ACTION:
                action_info['success'] = True
                action_info['message'] = "No action taken"
                
            elif action_type == ActionType.CREATE_LINE:
                success = self._action_create_line(param1, param2, param3)
                action_info['success'] = success
                action_info['message'] = f"Create line: {'success' if success else 'failed'}"
                
            elif action_type == ActionType.EXTEND_LINE:
                success = self._action_extend_line(param1, param2, param3)
                action_info['success'] = success
                action_info['message'] = f"Extend line: {'success' if success else 'failed'}"
                
            elif action_type == ActionType.ADD_TRAIN:
                success = self._action_add_train(param1)
                action_info['success'] = success
                action_info['message'] = f"Add train: {'success' if success else 'failed'}"
                
            elif action_type == ActionType.ADD_CARRIAGE:
                success = self._action_add_carriage(param1, param2)
                action_info['success'] = success
                action_info['message'] = f"Add carriage: {'success' if success else 'failed'}"
                
            elif action_type == ActionType.CONVERT_TO_LOOP:
                success = self._action_convert_to_loop(param1)
                action_info['success'] = success
                action_info['message'] = f"Convert to loop: {'success' if success else 'failed'}"
                
            elif action_type == ActionType.CONVERT_TO_LINEAR:
                success = self._action_convert_to_linear(param1)
                action_info['success'] = success
                action_info['message'] = f"Convert to linear: {'success' if success else 'failed'}"
                
            elif action_type == ActionType.ADD_BRIDGE_TUNNEL:
                success = self._action_add_bridge_tunnel(param1, param2, param3)
                action_info['success'] = success
                action_info['message'] = f"Add bridge/tunnel: {'success' if success else 'failed'}"
                
            else:
                action_info['message'] = f"Action {action_type.name} not implemented"
                
        except Exception as e:
            logger.warning(f"Error executing action {action_type.name}: {e}")
            action_info['message'] = f"Error: {str(e)}"
        
        return action_info
    
    def _action_create_line(self, station1: int, station2: int, color_idx: int) -> bool:
        """Create a new line connecting two stations."""
        if station1 >= len(self.game.stations) or station2 >= len(self.game.stations):
            return False
        
        if station1 == station2:
            return False
        
        # Get available color
        colors = list(LineColor)
        used_colors = {line.color for line in self.game.lines.values()}
        available_colors = [c for c in colors if c not in used_colors]
        
        if not available_colors:
            return False
        
        color_idx = min(color_idx, len(available_colors) - 1)
        color = available_colors[color_idx]
        
        line_id = self.game.create_line(color, [station1, station2])
        return line_id is not None
    
    def _action_extend_line(self, line_id: int, station_id: int, position: int) -> bool:
        """Extend a line with a new station."""
        if line_id not in self.game.lines:
            return False
        
        if station_id >= len(self.game.stations):
            return False
        
        # Convert position to valid range or None for append
        if position >= len(self.game.lines[line_id].stations):
            position = None
        
        return self.game.extend_line(line_id, station_id, position)
    
    def _action_add_train(self, line_id: int) -> bool:
        """Add a train to a line."""
        if line_id not in self.game.lines:
            return False
        
        return self.game.add_train_to_line(line_id)
    
    def _action_add_carriage(self, line_id: int, train_id: int) -> bool:
        """Add a carriage to a train."""
        if line_id not in self.game.lines:
            return False
        
        return self.game.add_carriage_to_train(line_id, train_id)
    
    def _action_convert_to_loop(self, line_id: int) -> bool:
        """Convert a line to loop configuration."""
        if line_id not in self.game.lines:
            return False
        
        return self.game.lines[line_id].convert_to_loop()
    
    def _action_convert_to_linear(self, line_id: int) -> bool:
        """Convert a line to linear configuration."""
        if line_id not in self.game.lines:
            return False
        
        return self.game.lines[line_id].convert_to_linear()
    
    def _action_add_bridge_tunnel(self, line_id: int, station1: int, station2: int) -> bool:
        """Add a bridge or tunnel to a line."""
        if line_id not in self.game.lines:
            return False
        
        if (station1 >= len(self.game.stations) or 
            station2 >= len(self.game.stations)):
            return False
        
        # Determine if bridge or tunnel is needed based on water bodies
        # For simplicity, always create bridge for now
        return self.game.lines[line_id].add_bridge_tunnel(station1, station2, is_bridge=True)
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        if self.config.observation_type == "vector":
            return self._get_vector_observation()
        elif self.config.observation_type == "image":
            return self._get_image_observation()
        else:
            raise NotImplementedError()
    
    def _get_vector_observation(self) -> np.ndarray:
        """Get vector-based observation."""
        game_state = self.game.get_detailed_state()
        
        # Global state
        global_obs = self.game.get_game_state_vector()
        
        # Station observations
        max_stations = 20
        station_obs = np.zeros(max_stations * 15, dtype=np.float32)
        
        stations = game_state.get('stations', {})
        for i, (station_id, station_data) in enumerate(stations.items()):
            if i >= max_stations:
                break
            start_idx = i * 15
            if isinstance(station_data, list) and len(station_data) >= 15:
                station_obs[start_idx:start_idx + 15] = station_data[:15]
        
        # Line observations
        max_lines = 10
        line_obs = np.zeros(max_lines * 10, dtype=np.float32)
        
        lines = game_state.get('lines', {})
        for i, (line_id, line_data) in enumerate(lines.items()):
            if i >= max_lines:
                break
            start_idx = i * 10
            # Extract key line metrics
            line_vector = [
                len(line_data.get('stations', [])) / 10.0,  # Normalized station count
                line_data.get('train_count', 0) / 4.0,      # Normalized train count
                1.0 if line_data.get('line_type') == 'loop' else 0.0,
                1.0 if line_data.get('has_bridge') else 0.0,
                1.0 if line_data.get('has_tunnel') else 0.0,
                line_data.get('stats', {}).get('average_utilization', 0.0),
                line_data.get('stats', {}).get('efficiency_score', 0.0),
                1.0 if line_data.get('is_active') else 0.0,
                line_data.get('total_length', 0.0) / 1000.0,  # Normalized length
                0.0  # Padding
            ]
            line_obs[start_idx:start_idx + 10] = line_vector[:10]
        
        # Train observations
        max_trains = 20
        train_obs = np.zeros(max_trains * 12, dtype=np.float32)
        
        train_count = 0
        for line_data in lines.values():
            trains = line_data.get('trains', [])
            for train_data in trains:
                if train_count >= max_trains:
                    break
                start_idx = train_count * 12
                
                # Extract key train metrics
                train_vector = [
                    train_data.get('position', [0, 0])[0] / 800.0,  # Normalized x
                    train_data.get('position', [0, 0])[1] / 600.0,  # Normalized y
                    train_data.get('direction', 1.0),
                    train_data.get('passenger_count', 0) / train_data.get('capacity', 1),
                    train_data.get('utilization', 0.0),
                    1.0 if train_data.get('is_at_station') else 0.0,
                    1.0 if train_data.get('is_loading') else 0.0,
                    train_data.get('progress_to_target', 0.0),
                    train_data.get('carriages', 1) / 6.0,  # Normalized carriage count
                    train_data.get('stats', {}).get('avg_utilization', 0.0),
                    0.0,  # Padding
                    0.0   # Padding
                ]
                train_obs[start_idx:start_idx + 12] = train_vector[:12]
                train_count += 1
        
        # Combine all observations
        full_observation = np.concatenate([
            global_obs,
            station_obs,
            line_obs,
            train_obs
        ])
        
        # Ensure observation matches expected size
        expected_size = self.observation_space.shape[0]
        if len(full_observation) < expected_size:
            # Pad with zeros
            padding = np.zeros(expected_size - len(full_observation), dtype=np.float32)
            full_observation = np.concatenate([full_observation, padding])
        elif len(full_observation) > expected_size:
            # Truncate
            full_observation = full_observation[:expected_size]
        
        return full_observation.astype(np.float32)
    
    def _get_image_observation(self) -> np.ndarray:
        """Get image-based observation."""
        # This would require rendering the game state to an image
        # For now, return a placeholder
        return np.zeros((600, 800, 3), dtype=np.uint8)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary."""
        game_state = self.game.get_detailed_state()
        stats = game_state.get('statistics', {})
        
        return {
            'episode_step': self.episode_step,
            'game_time': self.game.current_time,
            'game_week': self.game.stats.current_week,
            'passengers_delivered': stats.get('passengers_delivered', 0),
            'passengers_spawned': stats.get('passengers_spawned', 0),
            'delivery_rate': stats.get('delivery_rate', 0.0),
            'average_satisfaction': stats.get('average_satisfaction', 0.0),
            'total_reward': sum(self.episode_rewards),
            'stations_count': len(self.game.stations),
            'lines_count': len(self.game.lines),
            'is_game_over': self.game.is_game_over()
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', 'ansi')
            
        Returns:
            Rendered output (depends on mode)
        """
        if mode == 'ansi':
            # Text representation
            game_state = self.game.get_detailed_state()
            stats = game_state.get('statistics', {})
            
            output = []
            output.append(f"Mini Metro - Week {self.game.stats.current_week}")
            output.append(f"Time: {self.game.current_time:.1f}s")
            output.append(f"Stations: {len(self.game.stations)}")
            output.append(f"Lines: {len(self.game.lines)}")
            output.append(f"Passengers delivered: {stats.get('passengers_delivered', 0)}")
            output.append(f"Delivery rate: {stats.get('delivery_rate', 0.0):.2f}")
            output.append(f"Satisfaction: {stats.get('average_satisfaction', 0.0):.2f}")
            
            return '\n'.join(output)
        
        elif mode in ['human', 'rgb_array']:
            # Visual rendering would require pygame renderer
            # For now, return placeholder
            if mode == 'rgb_array':
                return np.zeros((600, 800, 3), dtype=np.uint8)
            else:
                print(self.render('ansi'))
                return None
        
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")
    
    def close(self) -> None:
        """Clean up environment resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        
        logger.debug("Environment closed")
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions in current state.
        
        Returns:
            Boolean array indicating valid actions
        """
        # This would analyze the current game state and determine
        # which actions are valid (e.g., can't create line if no resources)
        # For now, return all actions as valid
        mask = np.ones(len(ActionType), dtype=bool)
        
        # Check resource constraints
        if self.game.used_resources.lines >= self.game.resources.lines:
            mask[ActionType.CREATE_LINE.value] = False
        
        if self.game.used_resources.trains >= self.game.resources.trains:
            mask[ActionType.ADD_TRAIN.value] = False
        
        if self.game.used_resources.carriages >= self.game.resources.carriages:
            mask[ActionType.ADD_CARRIAGE.value] = False
        
        return mask