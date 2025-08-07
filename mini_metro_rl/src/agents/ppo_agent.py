"""
Proximal Policy Optimization (PPO) agent implementation for Mini Metro RL.

Implements PPO with clipped objective, generalized advantage estimation,
and comprehensive logging for stable policy learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque
from dataclasses import dataclass

from ..utils.config import PPOConfig
from ..utils.helpers import Timer, PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class PPOExperience:
    """Experience tuple for PPO training."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float
    advantage: float = 0.0
    returns: float = 0.0


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO with shared features."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [512, 512, 256],
        activation: str = "relu"
    ):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # Shared feature layers
        self.shared_layers = nn.ModuleList()
        prev_size = state_dim
        
        for hidden_size in hidden_sizes[:-1]:
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            self.activation,
            nn.Linear(hidden_sizes[-1], action_dim)
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            self.activation,
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.debug(f"Initialized Actor-Critic network: state_dim={state_dim}, action_dim={action_dim}")
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for policy head (smaller weights)
        for module in self.actor_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        # Shared features
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        
        # Actor and critic outputs
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        
        return action_logits, state_value
    
    def get_action_and_value(
        self, 
        x: torch.Tensor, 
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value.
        
        Args:
            x: Input state tensor
            action_mask: Boolean mask of valid actions
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(x)
        
        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        # Create categorical distribution
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value."""
        _, value = self.forward(x)
        return value.squeeze(-1)
    
    def evaluate_actions(
        self, 
        x: torch.Tensor, 
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            x: Input state tensor
            actions: Actions to evaluate
            action_mask: Boolean mask of valid actions
            
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, values = self.forward(x)
        
        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        # Create categorical distribution
        action_dist = Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class PPOBuffer:
    """Experience buffer for PPO with GAE computation."""
    
    def __init__(self, capacity: int, state_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize PPO buffer.
        
        Args:
            capacity: Buffer capacity
            state_dim: State dimension
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        
        logger.debug(f"Initialized PPO buffer: capacity={capacity}")
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ) -> None:
        """Add experience to buffer."""
        if self.position >= self.capacity:
            raise ValueError("Buffer is full")
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.log_probs[self.position] = log_prob
        self.values[self.position] = value
        
        self.position += 1
        self.size = max(self.size, self.position)
    
    def compute_advantages_and_returns(self, last_value: float = 0.0) -> None:
        """Compute advantages using GAE and returns."""
        # Convert to numpy if needed
        rewards = self.rewards[:self.size]
        values = self.values[:self.size]
        dones = self.dones[:self.size]
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Compute returns
        returns = advantages + values[:self.size]
        
        # Store computed values
        self.advantages[:self.size] = advantages
        self.returns[:self.size] = returns
        
        logger.debug(f"Computed advantages and returns for {self.size} experiences")
    
    def get_batch_generator(self, batch_size: int) -> Any:
        """Get mini-batch generator for training."""
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield {
                'states': self.states[batch_indices],
                'actions': self.actions[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'values': self.values[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices]
            }
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.position = 0
        self.size = 0
    
    def __len__(self) -> int:
        """Get buffer size."""
        return self.size


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Features:
    - Clipped policy objective
    - Generalized Advantage Estimation (GAE)
    - Adaptive KL penalty (optional)
    - Value function clipping
    - Entropy regularization
    - Comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: PPOConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PPO configuration
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim,
            action_dim,
            config.network.hidden_sizes,
            config.network.activation
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )
        
        # Experience buffer
        self.buffer = PPOBuffer(
            config.n_steps,
            state_dim,
            config.gamma,
            config.gae_lambda
        )
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.update_count = 0
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"Initialized PPO agent on {self.device}")
        logger.info(f"Network architecture: {config.network.hidden_sizes}")
        logger.info(f"Training config: n_steps={config.n_steps}, "
                   f"batch_size={config.batch_size}, n_epochs={config.n_epochs}")
    
    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """
        Select action and get value estimate.
        
        Args:
            state: Current state
            action_mask: Boolean mask of valid actions
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if action_mask is not None:
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
        else:
            mask_tensor = None
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action_and_value(state_tensor, mask_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ) -> Optional[Dict[str, Any]]:
        """
        Process one environment step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value estimate of state
            
        Returns:
            Training metrics if update was performed, None otherwise
        """
        self.step_count += 1
        
        # Add experience to buffer
        self.buffer.add(state, action, reward, next_state, done, log_prob, value)
        
        # Check if buffer is full and ready for update
        if len(self.buffer) >= self.config.n_steps:
            # Get last value for GAE computation
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    last_value = self.network.get_value(next_state_tensor).item()
            else:
                last_value = 0.0
            
            # Update agent
            metrics = self._update(last_value)
            
            # Clear buffer
            self.buffer.clear()
            
            return metrics
        
        return None
    
    def _update(self, last_value: float = 0.0) -> Dict[str, Any]:
        """
        Update the policy and value function.
        
        Args:
            last_value: Value estimate of the last state
            
        Returns:
            Training metrics
        """
        # Compute advantages and returns
        self.buffer.compute_advantages_and_returns(last_value)
        
        # Normalize advantages
        if self.config.normalize_advantage:
            advantages = self.buffer.advantages[:self.buffer.size]
            self.buffer.advantages[:self.buffer.size] = (
                (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            )
        
        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_divergence = 0.0
        total_clipfrac = 0.0
        n_updates = 0
        
        # Multiple epochs of training
        for epoch in range(self.config.n_epochs):
            # Mini-batch training
            for batch in self.buffer.get_batch_generator(self.config.batch_size):
                # Convert to tensors
                states = torch.FloatTensor(batch['states']).to(self.device)
                actions = torch.LongTensor(batch['actions']).to(self.device)
                old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
                old_values = torch.FloatTensor(batch['values']).to(self.device)
                advantages = torch.FloatTensor(batch['advantages']).to(self.device)
                returns = torch.FloatTensor(batch['returns']).to(self.device)
                
                # Get current policy and value estimates
                new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)
                
                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip_range,
                    1 + self.config.clip_range
                )
                
                policy_loss1 = -advantages * ratio
                policy_loss2 = -advantages * clipped_ratio
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # Value loss (optionally clipped)
                if self.config.clip_range_vf is not None:
                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    value_loss1 = F.mse_loss(new_values, returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, returns)
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = F.mse_loss(new_values, returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    kl_div = (old_log_probs - new_log_probs).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_kl_divergence += kl_div.item()
                total_clipfrac += clipfrac.item()
                n_updates += 1
        
        self.update_count += 1
        
        # Calculate average metrics
        metrics = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'kl_divergence': total_kl_divergence / n_updates,
            'clip_fraction': total_clipfrac / n_updates,
            'update_count': self.update_count,
            'step_count': self.step_count
        }
        
        # Record performance metrics
        self.performance_monitor.record(**metrics)
        
        logger.debug(f"PPO update {self.update_count}: "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"kl_div={metrics['kl_divergence']:.4f}")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save agent state.
        
        Args:
            filepath: Path to save file
        """
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved PPO agent to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load agent state.
        
        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.update_count = checkpoint['update_count']
        
        logger.info(f"Loaded PPO agent from {filepath}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        metrics = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'buffer_size': len(self.buffer),
            'performance_summary': self.performance_monitor.summary()
        }
        
        return metrics
    
    def end_episode(self) -> None:
        """Called at the end of each episode."""
        self.episode_count += 1
        logger.debug(f"Completed episode {self.episode_count}")