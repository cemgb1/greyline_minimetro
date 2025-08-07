"""
Reward calculation module for Mini Metro RL environment.

Implements comprehensive reward functions that encourage efficient network design,
passenger satisfaction, and optimal resource utilization.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewardWeights:
    """Weights for different reward components."""
    passenger_delivery: float = 10.0      # Reward for delivering passengers
    passenger_satisfaction: float = 5.0   # Reward for passenger satisfaction
    efficiency: float = 3.0               # Reward for network efficiency
    resource_optimization: float = 2.0    # Reward for optimal resource use
    congestion_penalty: float = -15.0     # Penalty for station congestion
    passenger_loss_penalty: float = -20.0 # Penalty for lost passengers
    game_over_penalty: float = -100.0     # Penalty for game over
    time_bonus: float = 1.0               # Bonus for surviving longer


class RewardCalculator:
    """
    Calculates rewards for the Mini Metro RL environment.
    
    Implements a multi-objective reward system that balances:
    - Passenger delivery and satisfaction
    - Network efficiency and optimization
    - Resource utilization
    - Congestion avoidance
    - Long-term sustainability
    """
    
    def __init__(self, weights: RewardWeights = None):
        """
        Initialize the reward calculator.
        
        Args:
            weights: Custom reward weights (uses defaults if None)
        """
        self.weights = weights or RewardWeights()
        self.previous_stats: Dict[str, Any] = {}
        self.episode_start_time = 0.0
        
        logger.debug("Initialized reward calculator with weights: %s", self.weights)
    
    def reset(self, initial_time: float = 0.0) -> None:
        """
        Reset the reward calculator for a new episode.
        
        Args:
            initial_time: Starting time for the episode
        """
        self.previous_stats = {}
        self.episode_start_time = initial_time
        logger.debug("Reset reward calculator")
    
    def calculate_reward(
        self,
        game_state: Dict[str, Any],
        action_taken: Dict[str, Any],
        is_terminal: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the reward for the current state and action.
        
        Args:
            game_state: Current game state dictionary
            action_taken: Action that was taken
            is_terminal: Whether this is a terminal state
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        reward_components = {}
        
        # Extract current statistics
        current_stats = game_state.get('statistics', {})
        current_time = game_state.get('current_time', 0.0)
        
        # 1. Passenger delivery reward
        delivery_reward = self._calculate_delivery_reward(current_stats)
        reward_components['delivery'] = delivery_reward
        
        # 2. Passenger satisfaction reward
        satisfaction_reward = self._calculate_satisfaction_reward(current_stats)
        reward_components['satisfaction'] = satisfaction_reward
        
        # 3. Network efficiency reward
        efficiency_reward = self._calculate_efficiency_reward(game_state)
        reward_components['efficiency'] = efficiency_reward
        
        # 4. Resource optimization reward
        resource_reward = self._calculate_resource_reward(game_state)
        reward_components['resource_optimization'] = resource_reward
        
        # 5. Congestion penalty
        congestion_penalty = self._calculate_congestion_penalty(game_state)
        reward_components['congestion'] = congestion_penalty
        
        # 6. Passenger loss penalty
        loss_penalty = self._calculate_loss_penalty(current_stats)
        reward_components['passenger_loss'] = loss_penalty
        
        # 7. Time survival bonus
        time_bonus = self._calculate_time_bonus(current_time)
        reward_components['time_bonus'] = time_bonus
        
        # 8. Game over penalty
        if is_terminal and game_state.get('game_state') == 'game_over':
            game_over_penalty = self.weights.game_over_penalty
            reward_components['game_over'] = game_over_penalty
        else:
            reward_components['game_over'] = 0.0
        
        # 9. Action-specific rewards
        action_reward = self._calculate_action_reward(action_taken, game_state)
        reward_components['action'] = action_reward
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        # Update previous stats for next calculation
        self.previous_stats = current_stats.copy()
        
        # Log reward breakdown for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Reward breakdown: %s (Total: %.3f)", 
                        {k: f"{v:.3f}" for k, v in reward_components.items()}, 
                        total_reward)
        
        return total_reward, reward_components
    
    def _calculate_delivery_reward(self, current_stats: Dict[str, Any]) -> float:
        """Calculate reward for passenger deliveries."""
        current_delivered = current_stats.get('passengers_delivered', 0)
        previous_delivered = self.previous_stats.get('passengers_delivered', 0)
        
        new_deliveries = current_delivered - previous_delivered
        return new_deliveries * self.weights.passenger_delivery
    
    def _calculate_satisfaction_reward(self, current_stats: Dict[str, Any]) -> float:
        """Calculate reward based on passenger satisfaction."""
        satisfaction = current_stats.get('average_satisfaction', 0.0)
        return satisfaction * self.weights.passenger_satisfaction
    
    def _calculate_efficiency_reward(self, game_state: Dict[str, Any]) -> float:
        """Calculate reward for network efficiency."""
        lines = game_state.get('lines', {})
        if not lines:
            return 0.0
        
        total_efficiency = 0.0
        for line_data in lines.values():
            stats = line_data.get('stats', {})
            efficiency = stats.get('efficiency_score', 0.0)
            utilization = stats.get('average_utilization', 0.0)
            
            # Reward balanced utilization (not too high, not too low)
            optimal_utilization = 0.7
            utilization_score = 1.0 - abs(utilization - optimal_utilization)
            
            total_efficiency += (efficiency + utilization_score) / 2.0
        
        average_efficiency = total_efficiency / len(lines)
        return average_efficiency * self.weights.efficiency
    
    def _calculate_resource_reward(self, game_state: Dict[str, Any]) -> float:
        """Calculate reward for optimal resource utilization."""
        resources = game_state.get('resources', {})
        available = resources.get('available', {})
        used = resources.get('used', {})
        
        if not available or not used:
            return 0.0
        
        # Calculate utilization ratios
        utilization_scores = []
        for resource_type in ['lines', 'trains', 'carriages']:
            if resource_type in available and resource_type in used:
                total = available[resource_type]
                utilized = used[resource_type]
                
                if total > 0:
                    ratio = utilized / total
                    # Reward moderate utilization (not waste, but not overuse)
                    optimal_ratio = 0.8
                    score = 1.0 - abs(ratio - optimal_ratio)
                    utilization_scores.append(max(0.0, score))
        
        if utilization_scores:
            average_utilization = np.mean(utilization_scores)
            return average_utilization * self.weights.resource_optimization
        
        return 0.0
    
    def _calculate_congestion_penalty(self, game_state: Dict[str, Any]) -> float:
        """Calculate penalty for station congestion."""
        stations = game_state.get('stations', {})
        if not stations:
            return 0.0
        
        congestion_count = 0
        total_stations = len(stations)
        
        # Count overloaded stations (this would need station state info)
        # For now, use a simplified approach based on queue ratios
        for station_data in stations.values():
            # Station data would be state vector - need to decode it properly
            # This is simplified - in practice, you'd extract queue utilization
            # from the station state vector
            pass
        
        # Simplified penalty based on delivery rate
        current_stats = game_state.get('statistics', {})
        delivery_rate = current_stats.get('delivery_rate', 1.0)
        
        if delivery_rate < 0.8:  # Less than 80% delivery rate indicates congestion
            congestion_penalty = (0.8 - delivery_rate) * self.weights.congestion_penalty
            return congestion_penalty
        
        return 0.0
    
    def _calculate_loss_penalty(self, current_stats: Dict[str, Any]) -> float:
        """Calculate penalty for lost passengers."""
        current_lost = current_stats.get('passengers_lost', 0)
        previous_lost = self.previous_stats.get('passengers_lost', 0)
        
        new_losses = current_lost - previous_lost
        return new_losses * self.weights.passenger_loss_penalty
    
    def _calculate_time_bonus(self, current_time: float) -> float:
        """Calculate bonus for surviving longer."""
        survival_time = current_time - self.episode_start_time
        # Diminishing returns for very long survival
        time_score = np.log(1 + survival_time / 60.0)  # Log scale in minutes
        return time_score * self.weights.time_bonus
    
    def _calculate_action_reward(
        self, 
        action_taken: Dict[str, Any], 
        game_state: Dict[str, Any]
    ) -> float:
        """Calculate reward/penalty for specific actions taken."""
        action_type = action_taken.get('type', 'no_action')
        
        # Small rewards for constructive actions
        constructive_actions = {
            'create_line': 2.0,
            'extend_line': 1.5,
            'add_train': 1.0,
            'add_carriage': 0.5,
            'add_bridge_tunnel': 3.0,
        }
        
        if action_type in constructive_actions:
            # Check if action was successful
            if action_taken.get('success', False):
                return constructive_actions[action_type]
            else:
                # Small penalty for failed actions
                return -0.5
        
        # Small penalty for doing nothing when action is needed
        if action_type == 'no_action':
            current_stats = game_state.get('statistics', {})
            delivery_rate = current_stats.get('delivery_rate', 1.0)
            
            if delivery_rate < 0.7:  # Poor performance - action needed
                return -1.0
        
        return 0.0
    
    def calculate_shaped_reward(
        self,
        game_state: Dict[str, Any],
        action_taken: Dict[str, Any],
        next_game_state: Dict[str, Any],
        is_terminal: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate shaped reward considering state transitions.
        
        Args:
            game_state: Previous game state
            action_taken: Action that was taken
            next_game_state: Resulting game state
            is_terminal: Whether next state is terminal
            
        Returns:
            Tuple of (shaped_reward, reward_breakdown)
        """
        # Get base reward
        base_reward, components = self.calculate_reward(
            next_game_state, action_taken, is_terminal
        )
        
        # Add potential-based shaping
        shaping_bonus = self._calculate_potential_shaping(
            game_state, next_game_state
        )
        components['potential_shaping'] = shaping_bonus
        
        total_shaped_reward = base_reward + shaping_bonus
        
        return total_shaped_reward, components
    
    def _calculate_potential_shaping(
        self,
        prev_state: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Calculate potential-based reward shaping.
        
        This provides additional guidance without changing the optimal policy.
        """
        # Potential function based on network connectivity and efficiency
        prev_potential = self._calculate_state_potential(prev_state)
        next_potential = self._calculate_state_potential(next_state)
        
        # Shaping is the difference in potential
        gamma = 0.99  # Discount factor
        shaping = gamma * next_potential - prev_potential
        
        return shaping * 0.1  # Scale down shaping to avoid overwhelming base reward
    
    def _calculate_state_potential(self, game_state: Dict[str, Any]) -> float:
        """Calculate the potential value of a game state."""
        if not game_state:
            return 0.0
        
        # Factors that contribute to state potential:
        # 1. Network connectivity
        lines = game_state.get('lines', {})
        stations = game_state.get('stations', {})
        
        connectivity_score = 0.0
        if lines and stations:
            # Calculate how well connected the network is
            total_stations = len(stations)
            connected_stations = set()
            
            for line_data in lines.values():
                line_stations = line_data.get('stations', [])
                connected_stations.update(line_stations)
            
            connectivity_ratio = len(connected_stations) / max(1, total_stations)
            connectivity_score = connectivity_ratio
        
        # 2. Resource efficiency potential
        resources = game_state.get('resources', {})
        efficiency_potential = 0.0
        if resources:
            available = resources.get('available', {})
            used = resources.get('used', {})
            
            # Potential increases with available unused resources
            for resource_type in ['lines', 'trains', 'carriages']:
                if resource_type in available and resource_type in used:
                    unused = available[resource_type] - used[resource_type]
                    efficiency_potential += unused * 0.1
        
        # 3. Performance potential
        stats = game_state.get('statistics', {})
        performance_potential = stats.get('delivery_rate', 0.0)
        
        total_potential = connectivity_score + efficiency_potential + performance_potential
        return total_potential
    
    def get_reward_summary(self) -> Dict[str, float]:
        """Get summary of reward weights for analysis."""
        return {
            'passenger_delivery': self.weights.passenger_delivery,
            'passenger_satisfaction': self.weights.passenger_satisfaction,
            'efficiency': self.weights.efficiency,
            'resource_optimization': self.weights.resource_optimization,
            'congestion_penalty': self.weights.congestion_penalty,
            'passenger_loss_penalty': self.weights.passenger_loss_penalty,
            'game_over_penalty': self.weights.game_over_penalty,
            'time_bonus': self.weights.time_bonus,
        }