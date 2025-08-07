"""
Deep Q-Network (DQN) agent implementation for Mini Metro RL.

Features advanced DQN variants including Double DQN, Dueling DQN,
Prioritized Experience Replay, and Noisy Networks for robust learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import deque, namedtuple
import random
from dataclasses import dataclass
import math

from ..utils.config import DQNConfig
from ..utils.helpers import Timer, PerformanceMonitor

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 'priority'
])


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration in neural networks."""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        """
        Initialize noisy linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            sigma_init: Initial noise standard deviation
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise tensors."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [512, 512, 256],
        use_noisy: bool = False,
        activation: str = "relu"
    ):
        """
        Initialize Dueling DQN.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            use_noisy: Whether to use noisy linear layers
            activation: Activation function name
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_noisy = use_noisy
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # Linear layer type
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        
        # Shared feature layers
        self.feature_layers = nn.ModuleList()
        prev_size = state_dim
        
        for hidden_size in hidden_sizes[:-1]:
            self.feature_layers.append(LinearLayer(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Value and advantage streams
        self.value_stream = nn.Sequential(
            LinearLayer(prev_size, hidden_sizes[-1]),
            self.activation,
            LinearLayer(hidden_sizes[-1], 1)
        )
        
        self.advantage_stream = nn.Sequential(
            LinearLayer(prev_size, hidden_sizes[-1]),
            self.activation,
            LinearLayer(hidden_sizes[-1], action_dim)
        )
        
        logger.debug(f"Initialized Dueling DQN: state_dim={state_dim}, action_dim={action_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling architecture."""
        # Shared features
        for layer in self.feature_layers:
            x = self.activation(layer(x))
        
        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine streams: Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with importance sampling."""
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 100000
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
            beta_start: Initial importance sampling weight
            beta_steps: Steps to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.step_count = 0
        
        # Storage
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
        logger.debug(f"Initialized prioritized replay buffer: capacity={capacity}")
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to buffer."""
        # Calculate initial priority (max priority for new experiences)
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        experience = Experience(state, action, reward, next_state, done, max_priority)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Tuple of (experiences, indices, weights)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} experiences, need {batch_size}")
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        beta = self._get_beta()
        
        weights = (total * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize weights
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Add small epsilon to avoid zero priority
    
    def _get_beta(self) -> float:
        """Get current beta value (annealed from beta_start to 1.0)."""
        progress = min(self.step_count / self.beta_steps, 1.0)
        beta = self.beta_start + (1.0 - self.beta_start) * progress
        self.step_count += 1
        return beta
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)


class StandardReplayBuffer:
    """Standard experience replay buffer."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        logger.debug(f"Initialized standard replay buffer: capacity={capacity}")
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done, 1.0)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], None, None]:
        """
        Sample random batch.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Tuple of (experiences, None, None) for compatibility
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} experiences, need {batch_size}")
        
        experiences = random.sample(self.buffer, batch_size)
        return experiences, None, None
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with advanced features.
    
    Features:
    - Double DQN for reduced overestimation
    - Dueling DQN for better value estimation
    - Prioritized Experience Replay
    - Noisy Networks for exploration
    - Multi-step learning
    - Comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DQNConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: DQN configuration
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DuelingDQN(
            state_dim,
            action_dim,
            config.network.hidden_sizes,
            config.noisy_networks,
            config.network.activation
        ).to(self.device)
        
        self.target_network = DuelingDQN(
            state_dim,
            action_dim,
            config.network.hidden_sizes,
            config.noisy_networks,
            config.network.activation
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # Replay buffer
        if config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                config.buffer_size,
                alpha=0.6,
                beta_start=0.4,
                beta_steps=config.epsilon_decay_steps
            )
        else:
            self.replay_buffer = StandardReplayBuffer(config.buffer_size)
        
        # Exploration
        self.epsilon = config.epsilon_start
        self.epsilon_decay = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_steps
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.training_step = 0
        
        # Multi-step learning
        self.n_step = config.n_step
        self.n_step_buffer = deque(maxlen=config.n_step)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"Initialized DQN agent on {self.device}")
        logger.info(f"Network architecture: {config.network.hidden_sizes}")
        logger.info(f"Features: Double DQN={config.double_dqn}, "
                   f"Dueling={config.dueling_dqn}, "
                   f"Prioritized Replay={config.prioritized_replay}, "
                   f"Noisy Networks={config.noisy_networks}")
    
    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """
        Select action using epsilon-greedy or noisy networks.
        
        Args:
            state: Current state
            action_mask: Boolean mask of valid actions
            
        Returns:
            Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy exploration (unless using noisy networks)
        if not self.config.noisy_networks and random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return np.random.choice(valid_actions)
            else:
                return random.randrange(self.action_dim)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            # Apply action mask if provided
            if action_mask is not None:
                q_values = q_values.clone()
                q_values[0, ~torch.BoolTensor(action_mask)] = float('-inf')
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Process one environment step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.step_count += 1
        
        # Store in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Add to replay buffer if n-step buffer is full
        if len(self.n_step_buffer) == self.n_step:
            self._add_n_step_experience()
        
        # Training
        if (len(self.replay_buffer) >= self.config.min_buffer_size and
            self.step_count % self.config.train_frequency == 0):
            self._train()
        
        # Update target network
        if self.step_count % self.config.target_update_frequency == 0:
            self._update_target_network()
        
        # Decay epsilon
        if not self.config.noisy_networks:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon - self.epsilon_decay
            )
        
        # Reset noise in noisy networks
        if self.config.noisy_networks:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
    
    def _add_n_step_experience(self) -> None:
        """Add n-step experience to replay buffer."""
        # Calculate n-step return
        n_step_return = 0
        gamma = self.config.gamma
        
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            n_step_return += (gamma ** i) * reward
            if done:
                break
        
        # Get first and last states
        first_experience = self.n_step_buffer[0]
        last_experience = self.n_step_buffer[-1]
        
        # Add to replay buffer
        self.replay_buffer.add(
            first_experience[0],  # state
            first_experience[1],  # action
            n_step_return,        # n-step reward
            last_experience[3],   # next_state
            last_experience[4]    # done
        )
    
    def _train(self) -> None:
        """Train the Q-network."""
        try:
            # Sample batch
            experiences, indices, weights = self.replay_buffer.sample(self.config.batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
            actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
            dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q-values
            with torch.no_grad():
                if self.config.double_dqn:
                    # Double DQN: use main network to select action, target network to evaluate
                    next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_network(next_states).gather(1, next_actions)
                else:
                    # Standard DQN
                    next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                
                target_q_values = rewards.unsqueeze(1) + (
                    self.config.gamma ** self.n_step
                ) * next_q_values * (~dones).unsqueeze(1)
            
            # Calculate loss
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                # Prioritized replay: use importance sampling weights
                weights_tensor = torch.FloatTensor(weights).to(self.device)
                loss = (weights_tensor * F.mse_loss(
                    current_q_values.squeeze(),
                    target_q_values.squeeze(),
                    reduction='none'
                )).mean()
                
                # Update priorities
                td_errors = torch.abs(current_q_values.squeeze() - target_q_values.squeeze())
                priorities = td_errors.detach().cpu().numpy()
                self.replay_buffer.update_priorities(indices, priorities)
            else:
                # Standard replay
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values.squeeze())
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.q_network.parameters(),
                    self.config.gradient_clipping
                )
            
            self.optimizer.step()
            self.training_step += 1
            
            # Record performance metrics
            self.performance_monitor.record(
                loss=loss.item(),
                q_value_mean=current_q_values.mean().item(),
                epsilon=self.epsilon,
                buffer_size=len(self.replay_buffer)
            )
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
    
    def _update_target_network(self) -> None:
        """Update target network."""
        if self.config.soft_update_tau > 0:
            # Soft update
            for target_param, local_param in zip(
                self.target_network.parameters(),
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.config.soft_update_tau * local_param.data +
                    (1.0 - self.config.soft_update_tau) * target_param.data
                )
        else:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        logger.debug("Updated target network")
    
    def save(self, filepath: str) -> None:
        """
        Save agent state.
        
        Args:
            filepath: Path to save file
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved agent to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load agent state.
        
        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint['training_step']
        
        logger.info(f"Loaded agent from {filepath}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        metrics = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'performance_summary': self.performance_monitor.summary()
        }
        
        return metrics
    
    def end_episode(self) -> None:
        """Called at the end of each episode."""
        self.episode_count += 1
        
        # Clear n-step buffer
        self.n_step_buffer.clear()
        
        logger.debug(f"Completed episode {self.episode_count}")