"""
Main Mini Metro game engine.

Implements the complete Mini Metro game with accurate mechanics including
real-time simulation, passenger management, resource allocation, and game progression.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from collections import defaultdict

from .station import Station, StationType, StationCapacityLevel
from .train import Train
from .line import Line, LineColor, LineType
from .passenger import Passenger

logger = logging.getLogger(__name__)


class GameState(Enum):
    """Current state of the game."""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"


class GameDifficulty(Enum):
    """Game difficulty levels."""
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class GameResources:
    """Available game resources."""
    lines: int = 3
    trains: int = 4
    carriages: int = 8
    bridges: int = 2
    tunnels: int = 2
    interchanges: int = 3


@dataclass
class GameStats:
    """Game performance statistics."""
    total_passengers_spawned: int = 0
    total_passengers_delivered: int = 0
    total_passengers_lost: int = 0
    current_week: int = 1
    game_duration: float = 0.0
    max_simultaneous_passengers: int = 0
    lines_built: int = 0
    stations_connected: int = 0
    average_passenger_satisfaction: float = 0.0
    efficiency_scores: List[float] = field(default_factory=list)


class MiniMetroGame:
    """
    Complete Mini Metro game implementation with realistic mechanics.
    
    Features:
    - Accurate game physics and timing
    - Dynamic passenger spawning with weekly escalation
    - Resource management and limitations
    - Multiple maps and configurations
    - Comprehensive performance tracking
    - Real-time state updates
    """
    
    def __init__(
        self,
        map_name: str = "london",
        difficulty: GameDifficulty = GameDifficulty.NORMAL,
        real_time: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the Mini Metro game.
        
        Args:
            map_name: Name of the map to load
            difficulty: Game difficulty level
            real_time: Whether to run in real-time or step-based mode
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.map_name = map_name
        self.difficulty = difficulty
        self.real_time = real_time
        
        # Game state
        self.state = GameState.MENU
        self.current_time = 0.0
        self.last_update_time = time.time() if real_time else 0.0
        self.time_scale = 1.0
        
        # Game entities
        self.stations: Dict[int, Station] = {}
        self.lines: Dict[int, Line] = {}
        self.all_passengers: Dict[int, Passenger] = {}
        
        # Resources and constraints
        self.resources = GameResources()
        self.used_resources = GameResources(lines=0, trains=0, carriages=0, 
                                          bridges=0, tunnels=0, interchanges=0)
        
        # Game progression
        self.week_duration = 60.0  # 60 seconds per week
        self.passenger_spawn_rate = 1.0  # Base spawn rate
        self.week_escalation_factor = 1.2  # Difficulty multiplier per week
        
        # Statistics and performance
        self.stats = GameStats()
        self.passenger_id_counter = 0
        
        # Game over conditions
        self.max_station_overload_time = 30.0  # Game over if station overloaded for 30s
        self.overloaded_stations: Dict[int, float] = {}
        
        # Map configuration
        self.map_bounds = (800, 600)  # Map size in pixels
        self.water_bodies: List[Tuple[Tuple[float, float], float]] = []  # (center, radius)
        
        # Initialize map
        self._initialize_map()
        
        logger.info(f"Initialized Mini Metro game: {map_name} on {difficulty.value} difficulty")
    
    def _initialize_map(self) -> None:
        """Initialize the map with stations and geography."""
        # Create initial stations based on map
        initial_stations = self._get_initial_stations_for_map()
        
        for station_data in initial_stations:
            station = Station(
                station_id=station_data['id'],
                position=station_data['position'],
                station_type=station_data['type'],
                capacity_level=StationCapacityLevel.SMALL
            )
            self.stations[station.station_id] = station
        
        # Add water bodies for bridge/tunnel mechanics
        if self.map_name == "london":
            # Thames river
            self.water_bodies.append(((400, 350), 150))
        elif self.map_name == "paris":
            # Seine river
            self.water_bodies.append(((350, 300), 120))
    
    def _get_initial_stations_for_map(self) -> List[Dict]:
        """Get initial station configuration for the selected map."""
        if self.map_name == "london":
            return [
                {'id': 0, 'position': (200, 200), 'type': StationType.CIRCLE},
                {'id': 1, 'position': (400, 150), 'type': StationType.TRIANGLE},
                {'id': 2, 'position': (600, 250), 'type': StationType.SQUARE},
            ]
        elif self.map_name == "paris":
            return [
                {'id': 0, 'position': (150, 180), 'type': StationType.CIRCLE},
                {'id': 1, 'position': (350, 120), 'type': StationType.TRIANGLE},
                {'id': 2, 'position': (550, 200), 'type': StationType.SQUARE},
                {'id': 3, 'position': (300, 350), 'type': StationType.PENTAGON},
            ]
        else:
            # Default map
            return [
                {'id': 0, 'position': (200, 200), 'type': StationType.CIRCLE},
                {'id': 1, 'position': (400, 200), 'type': StationType.TRIANGLE},
                {'id': 2, 'position': (600, 200), 'type': StationType.SQUARE},
            ]
    
    def start_game(self) -> None:
        """Start a new game."""
        self.state = GameState.PLAYING
        self.current_time = 0.0
        self.last_update_time = time.time() if self.real_time else 0.0
        
        # Reset statistics
        self.stats = GameStats()
        self.passenger_id_counter = 0
        
        logger.info("Game started")
    
    def pause_game(self) -> None:
        """Pause the game."""
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
            logger.info("Game paused")
    
    def resume_game(self) -> None:
        """Resume the game."""
        if self.state == GameState.PAUSED:
            self.state = GameState.PLAYING
            self.last_update_time = time.time() if self.real_time else self.current_time
            logger.info("Game resumed")
    
    def update(self, dt: Optional[float] = None) -> None:
        """
        Update the game state.
        
        Args:
            dt: Time step (if None, calculated from real time)
        """
        if self.state != GameState.PLAYING:
            return
        
        # Calculate time step
        if dt is None:
            if self.real_time:
                current_real_time = time.time()
                dt = (current_real_time - self.last_update_time) * self.time_scale
                self.last_update_time = current_real_time
            else:
                dt = 1.0 / 60.0  # 60 FPS default
        
        self.current_time += dt
        
        # Update game week
        self._update_week_progression()
        
        # Update stations (passenger spawning, queue management)
        self._update_stations(dt)
        
        # Update lines and trains
        self._update_lines_and_trains(dt)
        
        # Check game over conditions
        self._check_game_over_conditions(dt)
        
        # Update statistics
        self._update_statistics()
        
        # Spawn new stations periodically
        self._maybe_spawn_new_station()
    
    def _update_week_progression(self) -> None:
        """Update game week and difficulty scaling."""
        current_week = int(self.current_time / self.week_duration) + 1
        
        if current_week > self.stats.current_week:
            self.stats.current_week = current_week
            
            # Increase spawn rate with weekly escalation
            self.passenger_spawn_rate *= self.week_escalation_factor
            
            # Add resources every few weeks
            if current_week % 3 == 0:
                self.resources.trains += 1
                self.resources.carriages += 2
                logger.info(f"Week {current_week}: Added resources (trains: {self.resources.trains}, "
                           f"carriages: {self.resources.carriages})")
            
            logger.info(f"Entered week {current_week}, spawn rate: {self.passenger_spawn_rate:.2f}")
    
    def _update_stations(self, dt: float) -> None:
        """Update all stations."""
        week_multiplier = self.passenger_spawn_rate
        
        for station in self.stations.values():
            station.update(self.current_time, week_multiplier)
    
    def _update_lines_and_trains(self, dt: float) -> None:
        """Update all lines and their trains."""
        station_positions = {sid: s.position for sid, s in self.stations.items()}
        station_types = {sid: s.station_type for sid, s in self.stations.items()}
        
        for line in self.lines.values():
            line.update(self.current_time, station_positions, station_types)
            
            # Handle train-station interactions
            for train in line.trains:
                if train.is_at_station and train.is_loading:
                    self._handle_train_station_interaction(train, line, station_types)
    
    def _handle_train_station_interaction(
        self,
        train: Train,
        line: Line,
        station_types: Dict[int, StationType]
    ) -> None:
        """Handle passenger loading/unloading at stations."""
        if train.current_station is None:
            return
        
        station = self.stations.get(train.current_station)
        if station is None:
            return
        
        # Unload passengers first
        unloaded_passengers = train.unload_passengers(
            station.station_type,
            self.current_time
        )
        
        # Update delivered passenger statistics
        for passenger in unloaded_passengers:
            self.stats.total_passengers_delivered += 1
        
        # Load new passengers
        reachable_types = line.get_reachable_station_types(station_types)
        available_passengers = list(station.passenger_queue)
        
        boarded_passengers = train.load_passengers(
            available_passengers,
            reachable_types,
            self.current_time
        )
        
        # Remove boarded passengers from station queue
        for passenger in boarded_passengers:
            if passenger in station.passenger_queue:
                station.passenger_queue.remove(passenger)
    
    def _check_game_over_conditions(self, dt: float) -> None:
        """Check for game over conditions."""
        # Check for overloaded stations
        for station_id, station in self.stations.items():
            if station.is_overloaded:
                if station_id not in self.overloaded_stations:
                    self.overloaded_stations[station_id] = self.current_time
                
                overload_duration = self.current_time - self.overloaded_stations[station_id]
                if overload_duration >= self.max_station_overload_time:
                    self._trigger_game_over(f"Station {station_id} overloaded for too long")
                    return
            else:
                self.overloaded_stations.pop(station_id, None)
    
    def _trigger_game_over(self, reason: str) -> None:
        """Trigger game over."""
        self.state = GameState.GAME_OVER
        logger.info(f"Game Over: {reason}")
        logger.info(f"Final stats - Week: {self.stats.current_week}, "
                   f"Passengers delivered: {self.stats.total_passengers_delivered}, "
                   f"Duration: {self.current_time:.1f}s")
    
    def _update_statistics(self) -> None:
        """Update game statistics."""
        self.stats.game_duration = self.current_time
        
        # Count total passengers
        total_passengers = sum(len(s.passenger_queue) for s in self.stations.values())
        total_passengers += sum(len(t.passengers) for line in self.lines.values() for t in line.trains)
        self.stats.max_simultaneous_passengers = max(
            self.stats.max_simultaneous_passengers,
            total_passengers
        )
        
        # Calculate average satisfaction
        all_passengers = list(self.all_passengers.values())
        if all_passengers:
            satisfactions = [p.get_satisfaction_score(self.current_time) for p in all_passengers]
            self.stats.average_passenger_satisfaction = np.mean(satisfactions)
    
    def _maybe_spawn_new_station(self) -> None:
        """Possibly spawn a new station based on game progression."""
        # Add new stations every 2 weeks
        if (self.stats.current_week > 1 and 
            self.stats.current_week % 2 == 0 and 
            len(self.stations) < 15):  # Max 15 stations
            
            # Check if we need to add a station this week
            expected_stations = 3 + (self.stats.current_week // 2)
            if len(self.stations) < expected_stations:
                self._spawn_new_station()
    
    def _spawn_new_station(self) -> None:
        """Spawn a new station on the map."""
        # Choose a random position avoiding existing stations and water
        attempts = 0
        while attempts < 50:  # Max attempts to find valid position
            x = np.random.uniform(50, self.map_bounds[0] - 50)
            y = np.random.uniform(50, self.map_bounds[1] - 50)
            position = (x, y)
            
            # Check distance from existing stations
            too_close = False
            for station in self.stations.values():
                distance = np.linalg.norm(np.array(position) - np.array(station.position))
                if distance < 80:  # Minimum distance between stations
                    too_close = True
                    break
            
            # Check water bodies
            in_water = False
            for water_center, water_radius in self.water_bodies:
                distance = np.linalg.norm(np.array(position) - np.array(water_center))
                if distance < water_radius:
                    in_water = True
                    break
            
            if not too_close and not in_water:
                # Choose station type based on game progression
                available_types = [StationType.CIRCLE, StationType.TRIANGLE, StationType.SQUARE]
                if self.stats.current_week >= 3:
                    available_types.append(StationType.PENTAGON)
                if self.stats.current_week >= 5:
                    available_types.extend([StationType.HEXAGON, StationType.DIAMOND])
                if self.stats.current_week >= 8:
                    available_types.extend([StationType.STAR, StationType.CROSS])
                
                station_type = np.random.choice(available_types)
                station_id = max(self.stations.keys()) + 1 if self.stations else 0
                
                station = Station(station_id, position, station_type)
                self.stations[station_id] = station
                
                logger.info(f"Spawned new {station_type.value} station {station_id} at {position}")
                return
            
            attempts += 1
        
        logger.warning("Failed to spawn new station after 50 attempts")
    
    # ============ GAME ACTIONS (for RL and player control) ============
    
    def create_line(self, color: LineColor, station_ids: List[int]) -> Optional[int]:
        """
        Create a new metro line.
        
        Args:
            color: Color for the new line
            station_ids: Initial stations for the line
            
        Returns:
            Line ID if successful, None otherwise
        """
        if self.used_resources.lines >= self.resources.lines:
            logger.warning("Cannot create line: no lines available")
            return None
        
        if len(station_ids) < 2:
            logger.warning("Cannot create line: need at least 2 stations")
            return None
        
        # Check if stations exist
        for station_id in station_ids:
            if station_id not in self.stations:
                logger.warning(f"Cannot create line: station {station_id} does not exist")
                return None
        
        # Check if color is available
        used_colors = {line.color for line in self.lines.values()}
        if color in used_colors:
            logger.warning(f"Cannot create line: color {color.value} already in use")
            return None
        
        line_id = len(self.lines)
        line = Line(line_id, color, station_ids.copy())
        self.lines[line_id] = line
        
        # Update station connections
        for station_id in station_ids:
            self.stations[station_id].add_line_connection(line_id)
        
        self.used_resources.lines += 1
        self.stats.lines_built += 1
        
        logger.info(f"Created line {line_id} ({color.value}) with stations {station_ids}")
        return line_id
    
    def extend_line(self, line_id: int, station_id: int, position: Optional[int] = None) -> bool:
        """
        Extend a line with a new station.
        
        Args:
            line_id: ID of the line to extend
            station_id: ID of the station to add
            position: Position in the line (None = append to end)
            
        Returns:
            True if successful
        """
        if line_id not in self.lines:
            logger.warning(f"Cannot extend line {line_id}: line does not exist")
            return False
        
        if station_id not in self.stations:
            logger.warning(f"Cannot extend line {line_id}: station {station_id} does not exist")
            return False
        
        line = self.lines[line_id]
        if line.add_station(station_id, position):
            self.stations[station_id].add_line_connection(line_id)
            logger.info(f"Extended line {line_id} with station {station_id}")
            return True
        
        return False
    
    def add_train_to_line(self, line_id: int) -> bool:
        """
        Add a train to a line.
        
        Args:
            line_id: ID of the line
            
        Returns:
            True if successful
        """
        if line_id not in self.lines:
            logger.warning(f"Cannot add train: line {line_id} does not exist")
            return False
        
        if self.used_resources.trains >= self.resources.trains:
            logger.warning("Cannot add train: no trains available")
            return False
        
        line = self.lines[line_id]
        station_positions = {sid: s.position for sid, s in self.stations.items()}
        
        train = line.add_train(station_positions)
        if train is not None:
            self.used_resources.trains += 1
            logger.info(f"Added train to line {line_id}")
            return True
        
        return False
    
    def add_carriage_to_train(self, line_id: int, train_id: int) -> bool:
        """
        Add a carriage to a train.
        
        Args:
            line_id: ID of the line
            train_id: ID of the train
            
        Returns:
            True if successful
        """
        if self.used_resources.carriages >= self.resources.carriages:
            logger.warning("Cannot add carriage: no carriages available")
            return False
        
        line = self.lines.get(line_id)
        if line is None:
            return False
        
        for train in line.trains:
            if train.train_id == train_id:
                if train.add_carriage():
                    self.used_resources.carriages += 1
                    logger.info(f"Added carriage to train {train_id} on line {line_id}")
                    return True
                break
        
        return False
    
    def get_game_state_vector(self) -> np.ndarray:
        """
        Get comprehensive game state for RL agents.
        
        Returns:
            Normalized state vector containing all game information
        """
        # Basic game state
        time_normalized = (self.current_time % self.week_duration) / self.week_duration
        week_normalized = min(self.stats.current_week / 20.0, 1.0)  # Cap at week 20
        
        # Resource utilization
        line_usage = self.used_resources.lines / max(1, self.resources.lines)
        train_usage = self.used_resources.trains / max(1, self.resources.trains)
        carriage_usage = self.used_resources.carriages / max(1, self.resources.carriages)
        
        # Station metrics
        total_queue_length = sum(len(s.passenger_queue) for s in self.stations.values())
        max_queue_capacity = sum(s.max_queue_size for s in self.stations.values())
        queue_utilization = total_queue_length / max(1, max_queue_capacity)
        
        overloaded_count = sum(1 for s in self.stations.values() if s.is_overloaded)
        overload_ratio = overloaded_count / max(1, len(self.stations))
        
        # Performance metrics
        delivery_rate = (self.stats.total_passengers_delivered / 
                        max(1, self.stats.total_passengers_spawned))
        satisfaction = self.stats.average_passenger_satisfaction
        
        # Compile state vector
        state_vector = np.array([
            time_normalized,
            week_normalized,
            line_usage,
            train_usage,
            carriage_usage,
            queue_utilization,
            overload_ratio,
            delivery_rate,
            satisfaction,
            len(self.stations) / 20.0,  # Normalized station count
            len(self.lines) / 10.0,     # Normalized line count
        ], dtype=np.float32)
        
        return state_vector
    
    def get_detailed_state(self) -> Dict[str, Any]:
        """
        Get detailed game state for visualization and analysis.
        
        Returns:
            Dictionary containing comprehensive game state
        """
        return {
            'game_state': self.state.value,
            'current_time': self.current_time,
            'current_week': self.stats.current_week,
            'map_name': self.map_name,
            'difficulty': self.difficulty.value,
            'resources': {
                'available': {
                    'lines': self.resources.lines,
                    'trains': self.resources.trains,
                    'carriages': self.resources.carriages,
                    'bridges': self.resources.bridges,
                    'tunnels': self.resources.tunnels,
                },
                'used': {
                    'lines': self.used_resources.lines,
                    'trains': self.used_resources.trains,
                    'carriages': self.used_resources.carriages,
                    'bridges': self.used_resources.bridges,
                    'tunnels': self.used_resources.tunnels,
                }
            },
            'stations': {
                str(sid): station.get_state_vector().tolist() 
                for sid, station in self.stations.items()
            },
            'lines': {
                str(lid): line.get_detailed_state() 
                for lid, line in self.lines.items()
            },
            'statistics': {
                'passengers_spawned': self.stats.total_passengers_spawned,
                'passengers_delivered': self.stats.total_passengers_delivered,
                'passengers_lost': self.stats.total_passengers_lost,
                'delivery_rate': (self.stats.total_passengers_delivered / 
                                max(1, self.stats.total_passengers_spawned)),
                'average_satisfaction': self.stats.average_passenger_satisfaction,
                'game_duration': self.stats.game_duration,
                'max_simultaneous_passengers': self.stats.max_simultaneous_passengers,
            },
            'water_bodies': self.water_bodies,
            'map_bounds': self.map_bounds,
        }
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.state == GameState.GAME_OVER
    
    def reset(self) -> None:
        """Reset the game to initial state."""
        self.__init__(self.map_name, self.difficulty, self.real_time)
        logger.info("Game reset")