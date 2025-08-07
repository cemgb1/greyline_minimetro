"""
Train module for Mini Metro game.

Implements train entities with realistic movement, capacity, loading/unloading,
and carriage management that matches the original game mechanics.
"""

from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
import numpy as np
import logging
from dataclasses import dataclass
from .passenger import Passenger
from .station import StationType

logger = logging.getLogger(__name__)


class TrainDirection(Enum):
    """Train movement direction on a line."""
    FORWARD = 1
    BACKWARD = -1


@dataclass
class TrainStats:
    """Statistics tracking for train performance."""
    total_distance_traveled: float = 0.0
    passengers_carried: int = 0
    stops_made: int = 0
    time_in_stations: float = 0.0
    time_moving: float = 0.0
    utilization_history: List[float] = None
    
    def __post_init__(self):
        if self.utilization_history is None:
            self.utilization_history = []


class Train:
    """
    Represents a train in the Mini Metro game.
    
    Features realistic movement mechanics, passenger management,
    carriage system, and performance tracking.
    """
    
    def __init__(
        self,
        train_id: int,
        line_id: int,
        initial_position: Tuple[float, float],
        carriages: int = 1
    ):
        """
        Initialize a train.
        
        Args:
            train_id: Unique identifier for the train
            line_id: ID of the line this train operates on
            initial_position: Starting (x, y) position
            carriages: Number of carriages (affects capacity)
        """
        self.train_id = train_id
        self.line_id = line_id
        self.position = initial_position
        self.carriages = carriages
        
        # Movement properties
        self.speed = 50.0  # Base speed units per second
        self.direction = TrainDirection.FORWARD
        self.target_station: Optional[int] = None
        self.current_station: Optional[int] = None
        self.progress_to_target = 0.0  # 0.0 to 1.0
        
        # Passenger management
        self.passengers: List[Passenger] = []
        self.max_capacity = self._calculate_capacity()
        
        # Station operations
        self.is_at_station = False
        self.station_arrival_time: Optional[float] = None
        self.loading_duration = 3.0  # Seconds to load/unload at station
        self.is_loading = False
        
        # Performance tracking
        self.stats = TrainStats()
        self.last_update_time: Optional[float] = None
        
        logger.debug(f"Created train {train_id} on line {line_id} with {carriages} carriages")
    
    def _calculate_capacity(self) -> int:
        """Calculate train capacity based on number of carriages."""
        # Each carriage holds 6 passengers
        return self.carriages * 6
    
    def update(
        self,
        current_time: float,
        line_stations: List[int],
        station_positions: Dict[int, Tuple[float, float]]
    ) -> None:
        """
        Update train state including movement and station operations.
        
        Args:
            current_time: Current game time
            line_stations: Ordered list of station IDs on the line
            station_positions: Mapping of station IDs to positions
        """
        if self.last_update_time is None:
            self.last_update_time = current_time
            return
        
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if self.is_at_station and self.is_loading:
            # Handle station loading/unloading operations
            self._handle_station_operations(current_time, dt)
        else:
            # Handle train movement
            self._handle_movement(dt, line_stations, station_positions)
        
        # Update statistics
        self._update_statistics(dt)
    
    def _handle_movement(
        self,
        dt: float,
        line_stations: List[int],
        station_positions: Dict[int, Tuple[float, float]]
    ) -> None:
        """Handle train movement along the line."""
        if not line_stations or len(line_stations) < 2:
            return
        
        # Determine current target station
        if self.target_station is None:
            self._set_next_target_station(line_stations)
        
        if self.target_station is None:
            return
        
        # Calculate movement
        target_pos = station_positions.get(self.target_station)
        if target_pos is None:
            return
        
        # Move towards target station
        distance_to_travel = self.speed * dt
        current_distance = np.linalg.norm(np.array(target_pos) - np.array(self.position))
        
        if current_distance <= distance_to_travel:
            # Reached target station
            self.position = target_pos
            self.current_station = self.target_station
            self.is_at_station = True
            self.station_arrival_time = self.last_update_time
            self.is_loading = True
            self.progress_to_target = 0.0
            
            # Update statistics
            self.stats.stops_made += 1
            self.stats.total_distance_traveled += current_distance
            
            logger.debug(f"Train {self.train_id} arrived at station {self.target_station}")
        else:
            # Move towards target
            direction_vector = (np.array(target_pos) - np.array(self.position)) / current_distance
            new_position = np.array(self.position) + direction_vector * distance_to_travel
            self.position = tuple(new_position)
            self.progress_to_target = 1.0 - (current_distance - distance_to_travel) / current_distance
            
            # Update statistics
            self.stats.total_distance_traveled += distance_to_travel
            self.stats.time_moving += dt
    
    def _set_next_target_station(self, line_stations: List[int]) -> None:
        """Determine the next target station based on line configuration."""
        if not line_stations:
            return
        
        # Find current position in line
        current_index = None
        if self.current_station in line_stations:
            current_index = line_stations.index(self.current_station)
        
        if current_index is None:
            # Not at a station, go to the nearest one
            self.target_station = line_stations[0]
            return
        
        # Determine next station based on direction
        if self.direction == TrainDirection.FORWARD:
            if current_index < len(line_stations) - 1:
                self.target_station = line_stations[current_index + 1]
            else:
                # Reached end, turn around
                self.direction = TrainDirection.BACKWARD
                if len(line_stations) > 1:
                    self.target_station = line_stations[current_index - 1]
        else:  # BACKWARD
            if current_index > 0:
                self.target_station = line_stations[current_index - 1]
            else:
                # Reached beginning, turn around
                self.direction = TrainDirection.FORWARD
                if len(line_stations) > 1:
                    self.target_station = line_stations[current_index + 1]
    
    def _handle_station_operations(self, current_time: float, dt: float) -> None:
        """Handle loading and unloading operations at stations."""
        if self.station_arrival_time is None:
            return
        
        time_at_station = current_time - self.station_arrival_time
        
        if time_at_station >= self.loading_duration:
            # Finished loading/unloading, ready to depart
            self.is_loading = False
            self.is_at_station = False
            self.station_arrival_time = None
            self.target_station = None  # Will be set in next update
            
            logger.debug(f"Train {self.train_id} departing from station {self.current_station}")
        else:
            # Still loading/unloading
            self.stats.time_in_stations += dt
    
    def load_passengers(
        self,
        available_passengers: List[Passenger],
        reachable_destinations: Set[StationType],
        current_time: float
    ) -> List[Passenger]:
        """
        Load passengers onto the train.
        
        Args:
            available_passengers: Passengers waiting at the station
            reachable_destinations: Station types reachable from this train's line
            current_time: Current game time
            
        Returns:
            List of passengers that boarded the train
        """
        if not self.is_at_station or not self.is_loading:
            return []
        
        available_capacity = self.max_capacity - len(self.passengers)
        if available_capacity <= 0:
            return []
        
        boarded_passengers = []
        
        # Load passengers whose destinations are reachable
        for passenger in available_passengers:
            if len(boarded_passengers) >= available_capacity:
                break
            
            if passenger.destination_type in reachable_destinations:
                passenger.board_train(self.line_id, current_time)
                self.passengers.append(passenger)
                boarded_passengers.append(passenger)
                
                # Update statistics
                self.stats.passengers_carried += 1
        
        if boarded_passengers:
            logger.debug(f"Train {self.train_id} loaded {len(boarded_passengers)} passengers")
        
        return boarded_passengers
    
    def unload_passengers(
        self,
        station_type: StationType,
        current_time: float
    ) -> List[Passenger]:
        """
        Unload passengers at the current station.
        
        Args:
            station_type: Type of the current station
            current_time: Current game time
            
        Returns:
            List of passengers that were unloaded
        """
        if not self.is_at_station or not self.is_loading:
            return []
        
        unloaded_passengers = []
        remaining_passengers = []
        
        for passenger in self.passengers:
            if passenger.is_at_correct_destination(station_type):
                # Passenger reached destination
                passenger.arrive_at_station(self.current_station, current_time)
                passenger.deliver_to_destination(current_time)
                unloaded_passengers.append(passenger)
            else:
                # Passenger continues journey
                passenger.arrive_at_station(self.current_station, current_time)
                remaining_passengers.append(passenger)
        
        self.passengers = remaining_passengers
        
        if unloaded_passengers:
            logger.debug(f"Train {self.train_id} unloaded {len(unloaded_passengers)} passengers")
        
        return unloaded_passengers
    
    def add_carriage(self) -> bool:
        """
        Add a carriage to the train.
        
        Returns:
            True if carriage was added successfully
        """
        max_carriages = 6  # Mini Metro limit
        if self.carriages < max_carriages:
            self.carriages += 1
            self.max_capacity = self._calculate_capacity()
            logger.info(f"Train {self.train_id} upgraded to {self.carriages} carriages")
            return True
        return False
    
    def get_utilization(self) -> float:
        """Get current passenger utilization ratio."""
        return len(self.passengers) / self.max_capacity if self.max_capacity > 0 else 0.0
    
    def _update_statistics(self, dt: float) -> None:
        """Update train performance statistics."""
        utilization = self.get_utilization()
        self.stats.utilization_history.append(utilization)
        
        # Keep only recent history to manage memory
        if len(self.stats.utilization_history) > 1000:
            self.stats.utilization_history = self.stats.utilization_history[-500:]
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get numerical state representation for RL agents.
        
        Returns:
            State vector containing:
            - Position (2D)
            - Direction
            - Speed
            - Passenger count/capacity
            - Station status
            - Target information
        """
        # Position (normalized to map bounds)
        pos_x, pos_y = self.position
        
        # Direction encoding
        direction_val = 1.0 if self.direction == TrainDirection.FORWARD else -1.0
        
        # Capacity and passenger metrics
        passenger_count = len(self.passengers)
        utilization = self.get_utilization()
        
        # Station status
        at_station = 1.0 if self.is_at_station else 0.0
        loading = 1.0 if self.is_loading else 0.0
        
        # Progress to target
        progress = self.progress_to_target
        
        # Performance metrics
        avg_utilization = (np.mean(self.stats.utilization_history) 
                          if self.stats.utilization_history else 0.0)
        
        state_vector = np.array([
            pos_x, pos_y,
            direction_val,
            self.speed / 100.0,  # Normalized speed
            passenger_count / self.max_capacity,
            utilization,
            at_station,
            loading,
            progress,
            avg_utilization,
            self.carriages / 6.0  # Normalized carriage count
        ], dtype=np.float32)
        
        return state_vector
    
    def get_detailed_state(self) -> Dict:
        """
        Get detailed state information for visualization and debugging.
        
        Returns:
            Dictionary containing comprehensive train state
        """
        passenger_destinations = [p.destination_type.value for p in self.passengers]
        
        return {
            'train_id': self.train_id,
            'line_id': self.line_id,
            'position': self.position,
            'carriages': self.carriages,
            'capacity': self.max_capacity,
            'passenger_count': len(self.passengers),
            'utilization': self.get_utilization(),
            'direction': self.direction.value,
            'target_station': self.target_station,
            'current_station': self.current_station,
            'is_at_station': self.is_at_station,
            'is_loading': self.is_loading,
            'progress_to_target': self.progress_to_target,
            'passenger_destinations': passenger_destinations,
            'stats': {
                'distance_traveled': self.stats.total_distance_traveled,
                'passengers_carried': self.stats.passengers_carried,
                'stops_made': self.stats.stops_made,
                'time_in_stations': self.stats.time_in_stations,
                'time_moving': self.stats.time_moving,
                'avg_utilization': (np.mean(self.stats.utilization_history) 
                                  if self.stats.utilization_history else 0.0)
            }
        }
    
    def __repr__(self) -> str:
        """String representation of the train."""
        return (f"Train(id={self.train_id}, line={self.line_id}, "
                f"pos={self.position}, passengers={len(self.passengers)}/{self.max_capacity}, "
                f"carriages={self.carriages})")