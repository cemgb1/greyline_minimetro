"""
Line module for Mini Metro game.

Implements metro lines with realistic mechanics including station ordering,
bidirectional/loop configurations, bridge/tunnel management, and line operations.
"""

from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import numpy as np
import logging
from dataclasses import dataclass
from .train import Train
from .station import StationType

logger = logging.getLogger(__name__)


class LineType(Enum):
    """Different line configurations."""
    LINEAR = "linear"
    LOOP = "loop"


class LineColor(Enum):
    """Available line colors in Mini Metro."""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    BROWN = "brown"
    PINK = "pink"
    GRAY = "gray"
    CYAN = "cyan"


@dataclass
class BridgeTunnel:
    """Represents a bridge or tunnel connection."""
    from_station: int
    to_station: int
    is_bridge: bool  # True for bridge, False for tunnel
    cost: int = 1  # Resource cost


@dataclass
class LineStats:
    """Performance statistics for a line."""
    passengers_carried: int = 0
    total_journey_time: float = 0.0
    average_utilization: float = 0.0
    congestion_events: int = 0
    efficiency_score: float = 0.0


class Line:
    """
    Represents a metro line with realistic Mini Metro mechanics.
    
    Features:
    - Station ordering and connectivity
    - Linear and loop configurations
    - Bridge and tunnel management
    - Train assignment and management
    - Performance tracking and optimization
    """
    
    def __init__(
        self,
        line_id: int,
        color: LineColor,
        initial_stations: List[int] = None
    ):
        """
        Initialize a metro line.
        
        Args:
            line_id: Unique identifier for the line
            color: Color of the line
            initial_stations: Initial list of station IDs (ordered)
        """
        self.line_id = line_id
        self.color = color
        self.stations = initial_stations or []
        self.line_type = LineType.LINEAR
        
        # Train management
        self.trains: List[Train] = []
        self.max_trains = 4  # Mini Metro limit per line
        
        # Infrastructure
        self.bridges_tunnels: List[BridgeTunnel] = []
        self.has_bridge = False
        self.has_tunnel = False
        
        # Line properties
        self.is_active = len(self.stations) >= 2
        self.total_length = 0.0
        self.station_distances: Dict[Tuple[int, int], float] = {}
        
        # Performance tracking
        self.stats = LineStats()
        
        # State tracking
        self.last_update_time: Optional[float] = None
        
        logger.debug(f"Created line {line_id} ({color.value}) with {len(self.stations)} stations")
    
    def add_station(self, station_id: int, position: Optional[int] = None) -> bool:
        """
        Add a station to the line.
        
        Args:
            station_id: ID of the station to add
            position: Position in the line (None = append to end)
            
        Returns:
            True if station was added successfully
        """
        if station_id in self.stations:
            logger.warning(f"Station {station_id} already on line {self.line_id}")
            return False
        
        if position is None:
            self.stations.append(station_id)
        else:
            position = max(0, min(position, len(self.stations)))
            self.stations.insert(position, station_id)
        
        self._update_line_properties()
        logger.debug(f"Added station {station_id} to line {self.line_id}")
        return True
    
    def remove_station(self, station_id: int) -> bool:
        """
        Remove a station from the line.
        
        Args:
            station_id: ID of the station to remove
            
        Returns:
            True if station was removed successfully
        """
        if station_id not in self.stations:
            logger.warning(f"Station {station_id} not on line {self.line_id}")
            return False
        
        self.stations.remove(station_id)
        self._update_line_properties()
        logger.debug(f"Removed station {station_id} from line {self.line_id}")
        return True
    
    def insert_station(self, station_id: int, after_station_id: int) -> bool:
        """
        Insert a station after another station on the line.
        
        Args:
            station_id: ID of the station to insert
            after_station_id: ID of the station to insert after
            
        Returns:
            True if station was inserted successfully
        """
        if station_id in self.stations:
            return False
        
        try:
            position = self.stations.index(after_station_id) + 1
            return self.add_station(station_id, position)
        except ValueError:
            logger.warning(f"Station {after_station_id} not found on line {self.line_id}")
            return False
    
    def reorder_stations(self, new_order: List[int]) -> bool:
        """
        Reorder stations on the line.
        
        Args:
            new_order: New ordered list of station IDs
            
        Returns:
            True if reordering was successful
        """
        # Validate that new order contains same stations
        if set(new_order) != set(self.stations):
            logger.warning(f"Invalid reorder: different stations")
            return False
        
        self.stations = new_order
        self._update_line_properties()
        logger.debug(f"Reordered stations on line {self.line_id}")
        return True
    
    def convert_to_loop(self) -> bool:
        """
        Convert line from linear to loop configuration.
        
        Returns:
            True if conversion was successful
        """
        if len(self.stations) < 3:
            logger.warning(f"Cannot convert line {self.line_id} to loop: need at least 3 stations")
            return False
        
        if self.line_type == LineType.LOOP:
            return True  # Already a loop
        
        self.line_type = LineType.LOOP
        self._update_line_properties()
        logger.info(f"Converted line {self.line_id} to loop")
        return True
    
    def convert_to_linear(self) -> bool:
        """
        Convert line from loop to linear configuration.
        
        Returns:
            True if conversion was successful
        """
        if self.line_type == LineType.LINEAR:
            return True  # Already linear
        
        self.line_type = LineType.LINEAR
        self._update_line_properties()
        logger.info(f"Converted line {self.line_id} to linear")
        return True
    
    def add_bridge_tunnel(
        self,
        from_station: int,
        to_station: int,
        is_bridge: bool
    ) -> bool:
        """
        Add a bridge or tunnel connection.
        
        Args:
            from_station: Source station ID
            to_station: Destination station ID
            is_bridge: True for bridge, False for tunnel
            
        Returns:
            True if bridge/tunnel was added successfully
        """
        if from_station not in self.stations or to_station not in self.stations:
            logger.warning(f"Cannot add bridge/tunnel: stations not on line")
            return False
        
        # Check if connection already exists
        for bt in self.bridges_tunnels:
            if ((bt.from_station == from_station and bt.to_station == to_station) or
                (bt.from_station == to_station and bt.to_station == from_station)):
                logger.warning(f"Bridge/tunnel already exists between stations")
                return False
        
        bridge_tunnel = BridgeTunnel(from_station, to_station, is_bridge)
        self.bridges_tunnels.append(bridge_tunnel)
        
        if is_bridge:
            self.has_bridge = True
        else:
            self.has_tunnel = True
        
        self._update_line_properties()
        logger.debug(f"Added {'bridge' if is_bridge else 'tunnel'} on line {self.line_id}")
        return True
    
    def add_train(self, station_positions: Dict[int, Tuple[float, float]]) -> Optional[Train]:
        """
        Add a new train to the line.
        
        Args:
            station_positions: Mapping of station IDs to positions
            
        Returns:
            The created train, or None if failed
        """
        if len(self.trains) >= self.max_trains:
            logger.warning(f"Line {self.line_id} already has maximum trains")
            return None
        
        if not self.stations:
            logger.warning(f"Cannot add train to line {self.line_id}: no stations")
            return None
        
        # Start train at first station
        start_station = self.stations[0]
        start_position = station_positions.get(start_station, (0.0, 0.0))
        
        train_id = len(self.trains)  # Simple ID assignment
        train = Train(train_id, self.line_id, start_position)
        train.current_station = start_station
        
        self.trains.append(train)
        logger.debug(f"Added train {train_id} to line {self.line_id}")
        return train
    
    def remove_train(self, train_id: int) -> bool:
        """
        Remove a train from the line.
        
        Args:
            train_id: ID of the train to remove
            
        Returns:
            True if train was removed successfully
        """
        for i, train in enumerate(self.trains):
            if train.train_id == train_id:
                self.trains.pop(i)
                logger.debug(f"Removed train {train_id} from line {self.line_id}")
                return True
        
        logger.warning(f"Train {train_id} not found on line {self.line_id}")
        return False
    
    def update(
        self,
        current_time: float,
        station_positions: Dict[int, Tuple[float, float]],
        station_types: Dict[int, StationType]
    ) -> None:
        """
        Update line state including train movement and operations.
        
        Args:
            current_time: Current game time
            station_positions: Mapping of station IDs to positions
            station_types: Mapping of station IDs to types
        """
        self.last_update_time = current_time
        
        # Update all trains on this line
        for train in self.trains:
            train.update(current_time, self.stations, station_positions)
        
        # Update line statistics
        self._update_statistics()
    
    def get_reachable_station_types(self, station_types: Dict[int, StationType]) -> Set[StationType]:
        """
        Get set of station types reachable by this line.
        
        Args:
            station_types: Mapping of station IDs to types
            
        Returns:
            Set of reachable station types
        """
        reachable_types = set()
        for station_id in self.stations:
            if station_id in station_types:
                reachable_types.add(station_types[station_id])
        return reachable_types
    
    def calculate_line_length(self, station_positions: Dict[int, Tuple[float, float]]) -> float:
        """
        Calculate total physical length of the line.
        
        Args:
            station_positions: Mapping of station IDs to positions
            
        Returns:
            Total line length
        """
        if len(self.stations) < 2:
            return 0.0
        
        total_length = 0.0
        
        for i in range(len(self.stations) - 1):
            station1 = self.stations[i]
            station2 = self.stations[i + 1]
            
            pos1 = station_positions.get(station1, (0.0, 0.0))
            pos2 = station_positions.get(station2, (0.0, 0.0))
            
            distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
            total_length += distance
            
            # Store distance for quick lookup
            self.station_distances[(station1, station2)] = distance
            self.station_distances[(station2, station1)] = distance
        
        # Add closing segment for loops
        if self.line_type == LineType.LOOP and len(self.stations) > 2:
            first_station = self.stations[0]
            last_station = self.stations[-1]
            
            pos1 = station_positions.get(first_station, (0.0, 0.0))
            pos2 = station_positions.get(last_station, (0.0, 0.0))
            
            distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
            total_length += distance
            
            self.station_distances[(first_station, last_station)] = distance
            self.station_distances[(last_station, first_station)] = distance
        
        return total_length
    
    def _update_line_properties(self) -> None:
        """Update line properties after modifications."""
        self.is_active = len(self.stations) >= 2
    
    def _update_statistics(self) -> None:
        """Update line performance statistics."""
        if not self.trains:
            return
        
        # Calculate average utilization
        total_utilization = sum(train.get_utilization() for train in self.trains)
        self.stats.average_utilization = total_utilization / len(self.trains)
        
        # Calculate passengers carried
        total_passengers = sum(train.stats.passengers_carried for train in self.trains)
        self.stats.passengers_carried = total_passengers
        
        # Calculate efficiency based on utilization and passenger throughput
        if self.stats.average_utilization > 0:
            self.stats.efficiency_score = min(1.0, self.stats.average_utilization * 1.2)
        else:
            self.stats.efficiency_score = 0.0
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get numerical state representation for RL agents.
        
        Returns:
            State vector containing line configuration and performance metrics
        """
        # Basic line properties
        station_count = len(self.stations) / 10.0  # Normalized (assume max 10 stations)
        train_count = len(self.trains) / self.max_trains
        is_loop = 1.0 if self.line_type == LineType.LOOP else 0.0
        
        # Infrastructure
        has_bridge = 1.0 if self.has_bridge else 0.0
        has_tunnel = 1.0 if self.has_tunnel else 0.0
        
        # Performance metrics
        avg_utilization = self.stats.average_utilization
        efficiency = self.stats.efficiency_score
        
        # Activity status
        is_active = 1.0 if self.is_active else 0.0
        
        state_vector = np.array([
            station_count,
            train_count,
            is_loop,
            has_bridge,
            has_tunnel,
            avg_utilization,
            efficiency,
            is_active,
            self.total_length / 1000.0  # Normalized length
        ], dtype=np.float32)
        
        return state_vector
    
    def get_detailed_state(self) -> Dict:
        """
        Get detailed state information for visualization and debugging.
        
        Returns:
            Dictionary containing comprehensive line state
        """
        train_states = [train.get_detailed_state() for train in self.trains]
        
        return {
            'line_id': self.line_id,
            'color': self.color.value,
            'stations': self.stations,
            'line_type': self.line_type.value,
            'train_count': len(self.trains),
            'max_trains': self.max_trains,
            'is_active': self.is_active,
            'total_length': self.total_length,
            'has_bridge': self.has_bridge,
            'has_tunnel': self.has_tunnel,
            'bridges_tunnels': [
                {
                    'from': bt.from_station,
                    'to': bt.to_station,
                    'type': 'bridge' if bt.is_bridge else 'tunnel'
                }
                for bt in self.bridges_tunnels
            ],
            'trains': train_states,
            'stats': {
                'passengers_carried': self.stats.passengers_carried,
                'average_utilization': self.stats.average_utilization,
                'efficiency_score': self.stats.efficiency_score,
                'congestion_events': self.stats.congestion_events
            }
        }
    
    def __repr__(self) -> str:
        """String representation of the line."""
        return (f"Line(id={self.line_id}, color={self.color.value}, "
                f"stations={len(self.stations)}, trains={len(self.trains)}, "
                f"type={self.line_type.value})")