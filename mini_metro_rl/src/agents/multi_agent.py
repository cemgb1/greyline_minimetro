"""
Multi-agent system for Mini Metro RL.

Implements multi-agent reinforcement learning where each metro line
is controlled by a separate agent, with coordination mechanisms.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from collections import defaultdict

from ..utils.config import MultiAgentConfig, DQNConfig, PPOConfig
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from ..utils.helpers import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about an individual agent."""
    agent_id: int
    line_id: int
    agent_type: str
    is_active: bool = True
    creation_step: int = 0


class MultiAgent:
    """
    Multi-agent system for Metro line control.
    
    Features:
    - Individual agents per metro line
    - Shared experience replay (optional)
    - Centralized critic (optional)
    - Agent coordination mechanisms
    - Dynamic agent creation/removal
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: MultiAgentConfig,
        agent_configs: Dict[str, Union[DQNConfig, PPOConfig]],
        device: Optional[torch.device] = None
    ):
        """
        Initialize multi-agent system.
        
        Args:
            state_dim: Dimension of individual agent state space
            action_dim: Dimension of individual agent action space
            config: Multi-agent configuration
            agent_configs: Configuration for each agent type
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.agent_configs = agent_configs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Agent management
        self.agents: Dict[int, Union[DQNAgent, PPOAgent]] = {}
        self.agent_info: Dict[int, AgentInfo] = {}
        self.line_to_agent: Dict[int, int] = {}  # line_id -> agent_id
        self.next_agent_id = 0
        
        # Global state tracking
        self.step_count = 0
        self.episode_count = 0
        
        # Shared experience (if enabled)
        self.shared_experiences = []
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Communication (if enabled)
        if config.enable_communication:
            self.communication_network = self._create_communication_network()
        else:
            self.communication_network = None
        
        logger.info(f"Initialized multi-agent system with {config.agent_type} agents")
        logger.info(f"Max agents: {config.num_agents}, "
                   f"Shared experience: {config.shared_experience}, "
                   f"Centralized critic: {config.centralized_critic}")
    
    def _create_communication_network(self) -> torch.nn.Module:
        """Create communication network for agent coordination."""
        # Simple communication network
        comm_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.config.communication_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.communication_dim, self.config.communication_dim)
        ).to(self.device)
        
        return comm_net
    
    def create_agent(self, line_id: int) -> int:
        """
        Create a new agent for a metro line.
        
        Args:
            line_id: ID of the metro line
            
        Returns:
            ID of the created agent
        """
        if len(self.agents) >= self.config.num_agents:
            logger.warning(f"Cannot create agent: maximum number ({self.config.num_agents}) reached")
            return -1
        
        if line_id in self.line_to_agent:
            logger.warning(f"Agent already exists for line {line_id}")
            return self.line_to_agent[line_id]
        
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        
        # Create agent based on type
        if self.config.agent_type == "dqn":
            agent_config = self.agent_configs.get("dqn")
            if agent_config is None:
                raise ValueError("DQN config not provided")
            agent = DQNAgent(self.state_dim, self.action_dim, agent_config, self.device)
        
        elif self.config.agent_type == "ppo":
            agent_config = self.agent_configs.get("ppo")
            if agent_config is None:
                raise ValueError("PPO config not provided")
            agent = PPOAgent(self.state_dim, self.action_dim, agent_config, self.device)
        
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")
        
        # Store agent
        self.agents[agent_id] = agent
        self.agent_info[agent_id] = AgentInfo(
            agent_id=agent_id,
            line_id=line_id,
            agent_type=self.config.agent_type,
            creation_step=self.step_count
        )
        self.line_to_agent[line_id] = agent_id
        
        logger.info(f"Created agent {agent_id} for line {line_id}")
        return agent_id
    
    def remove_agent(self, line_id: int) -> bool:
        """
        Remove agent for a metro line.
        
        Args:
            line_id: ID of the metro line
            
        Returns:
            True if agent was removed successfully
        """
        if line_id not in self.line_to_agent:
            logger.warning(f"No agent found for line {line_id}")
            return False
        
        agent_id = self.line_to_agent[line_id]
        
        # Remove from all dictionaries
        del self.agents[agent_id]
        del self.agent_info[agent_id]
        del self.line_to_agent[line_id]
        
        logger.info(f"Removed agent {agent_id} for line {line_id}")
        return True
    
    def act(
        self,
        observations: Dict[int, np.ndarray],
        action_masks: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get actions from all agents.
        
        Args:
            observations: Dict mapping line_id to observation
            action_masks: Dict mapping line_id to action mask
            
        Returns:
            Dict mapping line_id to action info
        """
        actions = {}
        
        # Process communication if enabled
        if self.config.enable_communication and self.communication_network is not None:
            comm_messages = self._process_communication(observations)
        else:
            comm_messages = {}
        
        for line_id, observation in observations.items():
            if line_id not in self.line_to_agent:
                # No agent for this line, skip or create one
                logger.debug(f"No agent for line {line_id}, skipping")
                continue
            
            agent_id = self.line_to_agent[line_id]
            agent = self.agents[agent_id]
            
            # Add communication message if available
            if line_id in comm_messages:
                # Concatenate observation with communication message
                observation = np.concatenate([observation, comm_messages[line_id]])
            
            # Get action mask
            action_mask = action_masks.get(line_id) if action_masks else None
            
            # Get action from agent
            if isinstance(agent, DQNAgent):
                action = agent.act(observation, action_mask)
                actions[line_id] = {
                    'action': action,
                    'agent_id': agent_id,
                    'agent_type': 'dqn'
                }
            
            elif isinstance(agent, PPOAgent):
                action, log_prob, value = agent.act(observation, action_mask)
                actions[line_id] = {
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'agent_id': agent_id,
                    'agent_type': 'ppo'
                }
        
        return actions
    
    def step(
        self,
        observations: Dict[int, np.ndarray],
        actions: Dict[int, Dict[str, Any]],
        rewards: Dict[int, float],
        next_observations: Dict[int, np.ndarray],
        dones: Dict[int, bool]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process environment step for all agents.
        
        Args:
            observations: Previous observations
            actions: Actions taken
            rewards: Rewards received
            next_observations: New observations
            dones: Done flags
            
        Returns:
            Dict with training metrics for each line
        """
        self.step_count += 1
        metrics = {}
        
        for line_id in observations.keys():
            if line_id not in self.line_to_agent:
                continue
            
            agent_id = self.line_to_agent[line_id]
            agent = self.agents[agent_id]
            
            observation = observations[line_id]
            action_info = actions.get(line_id, {})
            reward = rewards.get(line_id, 0.0)
            next_observation = next_observations[line_id]
            done = dones.get(line_id, False)
            
            # Store experience for shared replay if enabled
            if self.config.shared_experience:
                self.shared_experiences.append({
                    'state': observation,
                    'action': action_info.get('action'),
                    'reward': reward,
                    'next_state': next_observation,
                    'done': done,
                    'agent_id': agent_id,
                    'line_id': line_id
                })
            
            # Update agent
            if isinstance(agent, DQNAgent):
                agent.step(
                    observation,
                    action_info.get('action', 0),
                    reward,
                    next_observation,
                    done
                )
                
            elif isinstance(agent, PPOAgent):
                agent_metrics = agent.step(
                    observation,
                    action_info.get('action', 0),
                    reward,
                    next_observation,
                    done,
                    action_info.get('log_prob', 0.0),
                    action_info.get('value', 0.0)
                )
                
                if agent_metrics:
                    metrics[line_id] = agent_metrics
        
        # Process shared experience if enabled
        if self.config.shared_experience and len(self.shared_experiences) > 1000:
            self._process_shared_experience()
        
        return metrics
    
    def _process_communication(self, observations: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Process communication between agents.
        
        Args:
            observations: Current observations
            
        Returns:
            Dict mapping line_id to communication message
        """
        if not self.config.enable_communication or self.communication_network is None:
            return {}
        
        comm_messages = {}
        
        # Generate communication messages
        all_messages = []
        line_ids = []
        
        for line_id, observation in observations.items():
            if line_id in self.line_to_agent:
                obs_tensor = torch.FloatTensor(observation).to(self.device)
                with torch.no_grad():
                    message = self.communication_network(obs_tensor)
                all_messages.append(message)
                line_ids.append(line_id)
        
        if not all_messages:
            return comm_messages
        
        # Aggregate messages (simple average for now)
        if len(all_messages) > 1:
            aggregated_message = torch.stack(all_messages).mean(dim=0)
        else:
            aggregated_message = all_messages[0]
        
        # Broadcast aggregated message to all agents
        aggregated_np = aggregated_message.cpu().numpy()
        for line_id in line_ids:
            comm_messages[line_id] = aggregated_np
        
        return comm_messages
    
    def _process_shared_experience(self) -> None:
        """Process shared experience replay."""
        # For now, just clear the buffer
        # In a full implementation, you would train a shared network
        # or distribute experiences among agents
        self.shared_experiences.clear()
        logger.debug("Processed shared experience")
    
    def end_episode(self) -> None:
        """Called at the end of each episode."""
        self.episode_count += 1
        
        # End episode for all agents
        for agent in self.agents.values():
            agent.end_episode()
        
        # Clear shared experience buffer
        if self.config.shared_experience:
            self.shared_experiences.clear()
        
        logger.debug(f"Multi-agent episode {self.episode_count} ended")
    
    def save(self, filepath_prefix: str) -> None:
        """
        Save all agents.
        
        Args:
            filepath_prefix: Prefix for save files
        """
        for agent_id, agent in self.agents.items():
            filepath = f"{filepath_prefix}_agent_{agent_id}.pt"
            agent.save(filepath)
        
        # Save multi-agent state
        ma_state = {
            'agent_info': self.agent_info,
            'line_to_agent': self.line_to_agent,
            'next_agent_id': self.next_agent_id,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        torch.save(ma_state, f"{filepath_prefix}_multiagent.pt")
        logger.info(f"Saved multi-agent system to {filepath_prefix}")
    
    def load(self, filepath_prefix: str) -> None:
        """
        Load all agents.
        
        Args:
            filepath_prefix: Prefix for save files
        """
        # Load multi-agent state
        ma_state = torch.load(f"{filepath_prefix}_multiagent.pt", map_location=self.device)
        
        self.agent_info = ma_state['agent_info']
        self.line_to_agent = ma_state['line_to_agent']
        self.next_agent_id = ma_state['next_agent_id']
        self.step_count = ma_state['step_count']
        self.episode_count = ma_state['episode_count']
        
        # Load individual agents
        for agent_id, info in self.agent_info.items():
            # Create agent
            if info.agent_type == "dqn":
                agent_config = self.agent_configs.get("dqn")
                agent = DQNAgent(self.state_dim, self.action_dim, agent_config, self.device)
            elif info.agent_type == "ppo":
                agent_config = self.agent_configs.get("ppo")
                agent = PPOAgent(self.state_dim, self.action_dim, agent_config, self.device)
            else:
                continue
            
            # Load agent state
            filepath = f"{filepath_prefix}_agent_{agent_id}.pt"
            try:
                agent.load(filepath)
                self.agents[agent_id] = agent
            except FileNotFoundError:
                logger.warning(f"Agent file not found: {filepath}")
        
        logger.info(f"Loaded multi-agent system from {filepath_prefix}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from all agents."""
        metrics = {
            'num_agents': len(self.agents),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'active_lines': list(self.line_to_agent.keys()),
            'agent_metrics': {}
        }
        
        # Get metrics from individual agents
        for line_id, agent_id in self.line_to_agent.items():
            agent = self.agents[agent_id]
            agent_metrics = agent.get_metrics()
            metrics['agent_metrics'][line_id] = agent_metrics
        
        return metrics
    
    def get_agent_for_line(self, line_id: int) -> Optional[Union[DQNAgent, PPOAgent]]:
        """
        Get agent for a specific line.
        
        Args:
            line_id: ID of the metro line
            
        Returns:
            Agent instance or None if not found
        """
        if line_id not in self.line_to_agent:
            return None
        
        agent_id = self.line_to_agent[line_id]
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        Get list of all agents with their information.
        
        Returns:
            List of agent information dictionaries
        """
        agent_list = []
        
        for agent_id, info in self.agent_info.items():
            agent_dict = {
                'agent_id': agent_id,
                'line_id': info.line_id,
                'agent_type': info.agent_type,
                'is_active': info.is_active,
                'creation_step': info.creation_step,
                'current_step': self.step_count
            }
            agent_list.append(agent_dict)
        
        return agent_list
    
    def update_line_assignment(self, old_line_id: int, new_line_id: int) -> bool:
        """
        Update line assignment for an agent.
        
        Args:
            old_line_id: Current line ID
            new_line_id: New line ID
            
        Returns:
            True if update was successful
        """
        if old_line_id not in self.line_to_agent:
            logger.warning(f"No agent found for line {old_line_id}")
            return False
        
        if new_line_id in self.line_to_agent:
            logger.warning(f"Agent already exists for line {new_line_id}")
            return False
        
        agent_id = self.line_to_agent[old_line_id]
        
        # Update mappings
        del self.line_to_agent[old_line_id]
        self.line_to_agent[new_line_id] = agent_id
        
        # Update agent info
        self.agent_info[agent_id].line_id = new_line_id
        
        logger.info(f"Updated agent {agent_id} assignment from line {old_line_id} to {new_line_id}")
        return True