"""
Passenger module for Mini Metro game.

Implements passenger entities with destination types, priorities,
and travel behavior that matches the original game mechanics.
"""

from typing import Optional
from dataclasses import dataclass
import logging
from .station import StationType

logger = logging.getLogger(__name__)


@dataclass
class PassengerStats:
    """Statistics tracking for passenger journey."""
    total_waiting_time: float = 0.0
    total_travel_time: float = 0.0
    transfers_made: int = 0
    lines_used: list = None
    
    def __post_init__(self):
        if self.lines_used is None:
            self.lines_used = []


class Passenger:
    """
    Represents a passenger in the Mini Metro game.
    
    Passengers have specific destination types (shapes) and track their
    journey statistics for performance analysis and reward calculation.
    """
    
    def __init__(
        self,
        passenger_id: int,
        origin_station: int,
        destination_type: StationType,
        spawn_time: float,
        priority: int = 1
    ):
        """
        Initialize a passenger.
        
        Args:
            passenger_id: Unique identifier for the passenger
            origin_station: ID of the station where passenger spawned
            destination_type: Type of station passenger wants to reach
            spawn_time: Game time when passenger was created
            priority: Passenger priority (for future priority systems)
        """
        self.passenger_id = passenger_id
        self.origin_station = origin_station
        self.destination_type = destination_type
        self.spawn_time = spawn_time
        self.priority = priority
        
        # Journey tracking
        self.current_station: Optional[int] = origin_station
        self.current_line: Optional[int] = None
        self.board_time: Optional[float] = None
        self.arrival_time: Optional[float] = None
        
        # State tracking
        self.is_on_train = False
        self.is_delivered = False
        self.journey_path: list = [origin_station]  # Track path taken
        
        # Statistics
        self.stats = PassengerStats()
        
        logger.debug(f"Created passenger {passenger_id}: {origin_station} -> {destination_type}")
    
    def board_train(self, line_id: int, board_time: float) -> None:
        """
        Board a train on a specific line.
        
        Args:
            line_id: ID of the line the passenger is boarding
            board_time: Time when passenger boarded
        """
        self.is_on_train = True
        self.current_line = line_id
        self.board_time = board_time
        
        # Calculate waiting time
        waiting_time = board_time - self.spawn_time
        self.stats.total_waiting_time = waiting_time
        
        # Track line usage
        if line_id not in self.stats.lines_used:
            self.stats.lines_used.append(line_id)
            if len(self.stats.lines_used) > 1:
                self.stats.transfers_made += 1
        
        logger.debug(f"Passenger {self.passenger_id} boarded line {line_id} after waiting {waiting_time:.2f}s")
    
    def arrive_at_station(self, station_id: int, arrival_time: float) -> None:
        """
        Handle passenger arrival at a station.
        
        Args:
            station_id: ID of the station where passenger arrived
            arrival_time: Time of arrival
        """
        self.current_station = station_id
        self.journey_path.append(station_id)
        
        # If passenger was on a train, they're now off it
        if self.is_on_train:
            self.is_on_train = False
            self.current_line = None
        
        logger.debug(f"Passenger {self.passenger_id} arrived at station {station_id}")
    
    def deliver_to_destination(self, delivery_time: float) -> None:
        """
        Mark passenger as successfully delivered to destination.
        
        Args:
            delivery_time: Time when passenger reached destination
        """
        self.is_delivered = True
        self.arrival_time = delivery_time
        
        # Calculate total travel time
        if self.board_time is not None:
            self.stats.total_travel_time = delivery_time - self.board_time
        
        total_journey_time = delivery_time - self.spawn_time
        
        logger.info(f"Passenger {self.passenger_id} delivered in {total_journey_time:.2f}s "
                   f"(waiting: {self.stats.total_waiting_time:.2f}s, "
                   f"travel: {self.stats.total_travel_time:.2f}s, "
                   f"transfers: {self.stats.transfers_made})")
    
    def get_waiting_time(self, current_time: float) -> float:
        """
        Get current waiting time for passenger.
        
        Args:
            current_time: Current game time
            
        Returns:
            Time passenger has been waiting
        """
        if self.is_delivered:
            return self.stats.total_waiting_time
        elif self.board_time is not None:
            return self.stats.total_waiting_time
        else:
            return current_time - self.spawn_time
    
    def get_travel_time(self, current_time: float) -> float:
        """
        Get current travel time for passenger.
        
        Args:
            current_time: Current game time
            
        Returns:
            Time passenger has been traveling
        """
        if self.is_delivered:
            return self.stats.total_travel_time
        elif self.board_time is not None:
            return current_time - self.board_time
        else:
            return 0.0
    
    def get_total_journey_time(self, current_time: float) -> float:
        """
        Get total time since passenger spawned.
        
        Args:
            current_time: Current game time
            
        Returns:
            Total time since passenger was created
        """
        if self.is_delivered and self.arrival_time is not None:
            return self.arrival_time - self.spawn_time
        else:
            return current_time - self.spawn_time
    
    def is_at_correct_destination(self, station_type: StationType) -> bool:
        """
        Check if passenger is at their desired destination.
        
        Args:
            station_type: Type of the current station
            
        Returns:
            True if passenger is at correct destination
        """
        return station_type == self.destination_type
    
    def get_satisfaction_score(self, current_time: float) -> float:
        """
        Calculate passenger satisfaction score based on journey.
        
        Args:
            current_time: Current game time
            
        Returns:
            Satisfaction score between 0.0 and 1.0
        """
        if not self.is_delivered:
            # Penalize long waiting times
            waiting_time = self.get_waiting_time(current_time)
            if waiting_time > 60.0:  # More than 1 minute waiting
                return max(0.0, 1.0 - (waiting_time - 60.0) / 120.0)
            return 0.5  # Neutral while traveling
        
        # Calculate satisfaction based on journey efficiency
        total_time = self.stats.total_waiting_time + self.stats.total_travel_time
        
        # Base satisfaction
        satisfaction = 1.0
        
        # Penalize long waiting times
        if self.stats.total_waiting_time > 30.0:
            satisfaction -= (self.stats.total_waiting_time - 30.0) / 100.0
        
        # Penalize long travel times
        if self.stats.total_travel_time > 60.0:
            satisfaction -= (self.stats.total_travel_time - 60.0) / 150.0
        
        # Penalize transfers (but not too heavily)
        satisfaction -= self.stats.transfers_made * 0.1
        
        return max(0.0, min(1.0, satisfaction))
    
    def get_state_info(self) -> dict:
        """
        Get passenger state information for visualization and debugging.
        
        Returns:
            Dictionary containing passenger state information
        """
        return {
            'passenger_id': self.passenger_id,
            'origin_station': self.origin_station,
            'destination_type': self.destination_type.value,
            'current_station': self.current_station,
            'current_line': self.current_line,
            'is_on_train': self.is_on_train,
            'is_delivered': self.is_delivered,
            'spawn_time': self.spawn_time,
            'board_time': self.board_time,
            'arrival_time': self.arrival_time,
            'journey_path': self.journey_path,
            'stats': {
                'waiting_time': self.stats.total_waiting_time,
                'travel_time': self.stats.total_travel_time,
                'transfers': self.stats.transfers_made,
                'lines_used': self.stats.lines_used
            }
        }
    
    def __repr__(self) -> str:
        """String representation of the passenger."""
        status = "delivered" if self.is_delivered else ("on_train" if self.is_on_train else "waiting")
        return (f"Passenger(id={self.passenger_id}, "
                f"origin={self.origin_station}, "
                f"dest={self.destination_type.value}, "
                f"status={status})")