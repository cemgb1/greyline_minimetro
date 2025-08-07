"""
Station module for Mini Metro game.

Implements different station types with accurate game mechanics including
passenger spawning, queuing, capacity, and station-specific behaviors.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
import logging
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class StationType(Enum):
    """Different station types with unique shapes and behaviors."""
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    SQUARE = "square"
    PENTAGON = "pentagon"
    HEXAGON = "hexagon"
    DIAMOND = "diamond"
    STAR = "star"
    CROSS = "cross"


class StationCapacityLevel(Enum):
    """Station capacity upgrade levels."""
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


@dataclass
class PassengerDemand:
    """Represents passenger demand patterns for a station."""
    destination_type: StationType
    spawn_rate: float
    priority: int = 1


class Station:
    """
    Represents a Mini Metro station with realistic game mechanics.
    
    Features:
    - Different station types (shapes) with unique passenger generation
    - Capacity upgrades and queue management
    - Realistic passenger spawning patterns
    - Connection management for lines
    - Performance metrics tracking
    """
    
    def __init__(
        self,
        station_id: int,
        position: Tuple[float, float],
        station_type: StationType,
        capacity_level: StationCapacityLevel = StationCapacityLevel.SMALL
    ):
        """
        Initialize a station.
        
        Args:
            station_id: Unique identifier for the station
            position: (x, y) coordinates on the map
            station_type: Type/shape of the station
            capacity_level: Initial capacity upgrade level
        """
        self.station_id = station_id
        self.position = position
        self.station_type = station_type
        self.capacity_level = capacity_level
        
        # Passenger management
        self.passenger_queue: deque = deque()  # Queue of waiting passengers
        self.max_queue_size = self._calculate_max_capacity()
        self.total_passengers_spawned = 0
        self.total_passengers_served = 0
        
        # Connection management
        self.connected_lines: Set[int] = set()  # Line IDs connected to this station
        self.is_interchange = False
        
        # Demand patterns based on station type
        self.demand_patterns = self._initialize_demand_patterns()
        
        # Timing and state
        self.last_spawn_time = 0.0
        self.is_overloaded = False
        self.overload_start_time: Optional[float] = None
        
        # Statistics
        self.stats = {
            'total_waiting_time': 0.0,
            'max_queue_length': 0,
            'congestion_events': 0,
            'passengers_lost': 0
        }
        
        logger.debug(f"Created station {station_id} of type {station_type} at {position}")
    
    def _calculate_max_capacity(self) -> int:
        """Calculate maximum passenger capacity based on upgrade level."""
        base_capacity = {
            StationCapacityLevel.SMALL: 6,
            StationCapacityLevel.MEDIUM: 12,
            StationCapacityLevel.LARGE: 18
        }
        return base_capacity[self.capacity_level]
    
    def _initialize_demand_patterns(self) -> List[PassengerDemand]:
        """Initialize passenger demand patterns based on station type."""
        # Each station type generates passengers for different destinations
        # with realistic demand distributions
        
        base_patterns = {
            StationType.CIRCLE: [
                PassengerDemand(StationType.SQUARE, 0.3),
                PassengerDemand(StationType.TRIANGLE, 0.2),
            ],
            StationType.TRIANGLE: [
                PassengerDemand(StationType.CIRCLE, 0.2),
                PassengerDemand(StationType.SQUARE, 0.3),
                PassengerDemand(StationType.PENTAGON, 0.1),
            ],
            StationType.SQUARE: [
                PassengerDemand(StationType.CIRCLE, 0.3),
                PassengerDemand(StationType.TRIANGLE, 0.2),
                PassengerDemand(StationType.DIAMOND, 0.1),
            ],
            StationType.PENTAGON: [
                PassengerDemand(StationType.TRIANGLE, 0.2),
                PassengerDemand(StationType.HEXAGON, 0.1),
            ],
            StationType.HEXAGON: [
                PassengerDemand(StationType.PENTAGON, 0.1),
                PassengerDemand(StationType.STAR, 0.05),
            ],
            StationType.DIAMOND: [
                PassengerDemand(StationType.SQUARE, 0.15),
                PassengerDemand(StationType.CROSS, 0.05),
            ],
            StationType.STAR: [
                PassengerDemand(StationType.HEXAGON, 0.05),
            ],
            StationType.CROSS: [
                PassengerDemand(StationType.DIAMOND, 0.05),
            ]
        }
        
        return base_patterns.get(self.station_type, [])
    
    def update(self, current_time: float, week_multiplier: float = 1.0) -> None:
        """
        Update station state including passenger spawning.
        
        Args:
            current_time: Current game time
            week_multiplier: Multiplier for difficulty escalation
        """
        # Spawn new passengers based on demand patterns
        self._spawn_passengers(current_time, week_multiplier)
        
        # Check for overload conditions
        self._check_overload_status(current_time)
        
        # Update statistics
        self._update_statistics(current_time)
    
    def _spawn_passengers(self, current_time: float, week_multiplier: float) -> None:
        """Spawn passengers based on demand patterns and game progression."""
        if len(self.passenger_queue) >= self.max_queue_size:
            # Station is at capacity - passengers are lost
            self.stats['passengers_lost'] += 1
            logger.debug(f"Station {self.station_id} lost passenger - at capacity")
            return
        
        for demand in self.demand_patterns:
            # Calculate spawn probability with weekly escalation
            spawn_probability = demand.spawn_rate * week_multiplier
            
            # Add some randomness to spawning
            if np.random.random() < spawn_probability:
                from .passenger import Passenger  # Avoid circular import
                passenger = Passenger(
                    passenger_id=self.total_passengers_spawned,
                    origin_station=self.station_id,
                    destination_type=demand.destination_type,
                    spawn_time=current_time,
                    priority=demand.priority
                )
                
                self.passenger_queue.append(passenger)
                self.total_passengers_spawned += 1
                logger.debug(f"Spawned passenger {passenger.passenger_id} at station {self.station_id}")
    
    def _check_overload_status(self, current_time: float) -> None:
        """Check if station is overloaded and update status."""
        queue_length = len(self.passenger_queue)
        
        # Station is overloaded if queue exceeds 80% of capacity
        overload_threshold = int(self.max_queue_size * 0.8)
        
        if queue_length >= overload_threshold and not self.is_overloaded:
            self.is_overloaded = True
            self.overload_start_time = current_time
            self.stats['congestion_events'] += 1
            logger.warning(f"Station {self.station_id} is overloaded: {queue_length}/{self.max_queue_size}")
        
        elif queue_length < overload_threshold and self.is_overloaded:
            self.is_overloaded = False
            self.overload_start_time = None
            logger.info(f"Station {self.station_id} overload resolved")
    
    def _update_statistics(self, current_time: float) -> None:
        """Update station performance statistics."""
        queue_length = len(self.passenger_queue)
        self.stats['max_queue_length'] = max(self.stats['max_queue_length'], queue_length)
        
        # Update waiting times for passengers in queue
        for passenger in self.passenger_queue:
            waiting_time = current_time - passenger.spawn_time
            self.stats['total_waiting_time'] += waiting_time
    
    def board_passengers(
        self, 
        destination_types: Set[StationType], 
        max_capacity: int,
        current_time: float
    ) -> List['Passenger']:
        """
        Board passengers onto a train.
        
        Args:
            destination_types: Set of station types reachable by this train
            max_capacity: Maximum number of passengers that can board
            current_time: Current game time
        
        Returns:
            List of passengers that boarded the train
        """
        boarded_passengers = []
        passengers_to_remove = []
        
        # Check each passenger in queue
        for passenger in self.passenger_queue:
            if len(boarded_passengers) >= max_capacity:
                break
                
            # Check if passenger's destination is reachable
            if passenger.destination_type in destination_types:
                boarded_passengers.append(passenger)
                passengers_to_remove.append(passenger)
                
                # Update passenger boarding time
                passenger.board_time = current_time
                self.total_passengers_served += 1
        
        # Remove boarded passengers from queue
        for passenger in passengers_to_remove:
            self.passenger_queue.remove(passenger)
        
        if boarded_passengers:
            logger.debug(f"Station {self.station_id}: {len(boarded_passengers)} passengers boarded")
        
        return boarded_passengers
    
    def upgrade_capacity(self) -> bool:
        """
        Upgrade station capacity to next level.
        
        Returns:
            True if upgrade was successful, False if already at max level
        """
        current_levels = list(StationCapacityLevel)
        current_index = current_levels.index(self.capacity_level)
        
        if current_index < len(current_levels) - 1:
            self.capacity_level = current_levels[current_index + 1]
            self.max_queue_size = self._calculate_max_capacity()
            logger.info(f"Station {self.station_id} upgraded to {self.capacity_level}")
            return True
        
        return False
    
    def add_line_connection(self, line_id: int) -> None:
        """Add a line connection to this station."""
        self.connected_lines.add(line_id)
        self.is_interchange = len(self.connected_lines) > 1
        logger.debug(f"Station {self.station_id} connected to line {line_id}")
    
    def remove_line_connection(self, line_id: int) -> None:
        """Remove a line connection from this station."""
        self.connected_lines.discard(line_id)
        self.is_interchange = len(self.connected_lines) > 1
        logger.debug(f"Station {self.station_id} disconnected from line {line_id}")
    
    def get_reachable_destinations(self) -> Set[StationType]:
        """Get set of station types reachable from this station."""
        # This would be calculated based on connected lines
        # For now, return all possible types (simplified)
        return set(StationType)
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get numerical state representation for RL agents.
        
        Returns:
            State vector containing:
            - Position (2D)
            - Station type (one-hot encoded)
            - Queue length ratio
            - Capacity level
            - Connection count
            - Overload status
            - Performance metrics
        """
        # Position (normalized to 0-1 range)
        pos_x, pos_y = self.position
        
        # Station type (one-hot encoding)
        station_types = list(StationType)
        type_encoding = [1.0 if st == self.station_type else 0.0 for st in station_types]
        
        # Queue and capacity metrics
        queue_ratio = len(self.passenger_queue) / self.max_queue_size
        capacity_level = self.capacity_level.value / 3.0  # Normalized
        
        # Connection metrics
        connection_count = len(self.connected_lines) / 5.0  # Assume max 5 lines
        interchange_status = 1.0 if self.is_interchange else 0.0
        overload_status = 1.0 if self.is_overloaded else 0.0
        
        # Performance metrics
        efficiency = (self.total_passengers_served / 
                     max(1, self.total_passengers_spawned))
        
        state_vector = np.array([
            pos_x, pos_y,
            *type_encoding,
            queue_ratio,
            capacity_level,
            connection_count,
            interchange_status,
            overload_status,
            efficiency
        ], dtype=np.float32)
        
        return state_vector
    
    def __repr__(self) -> str:
        """String representation of the station."""
        return (f"Station(id={self.station_id}, type={self.station_type}, "
                f"pos={self.position}, queue={len(self.passenger_queue)}/{self.max_queue_size})")