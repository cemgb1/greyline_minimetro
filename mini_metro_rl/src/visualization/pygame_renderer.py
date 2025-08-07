"""
Pygame-based renderer for Mini Metro RL visualization.

Provides real-time rendering of the game state with smooth animations,
train movement, passenger visualization, and performance metrics.
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import deque

from ..game.station import StationType
from ..game.line import LineColor
from ..utils.helpers import create_colormap, interpolate_positions

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """Configuration for pygame renderer."""
    width: int = 800
    height: int = 600
    fps: int = 30
    background_color: Tuple[int, int, int] = (20, 30, 40)
    
    # Station rendering
    station_radius: int = 12
    station_queue_width: int = 3
    show_passenger_destinations: bool = True
    show_station_queues: bool = True
    
    # Train rendering
    train_size: int = 8
    show_train_loads: bool = True
    train_speed_multiplier: float = 2.0
    
    # Line rendering
    line_width: int = 4
    
    # UI rendering
    show_performance_metrics: bool = True
    font_size: int = 16
    metric_panel_width: int = 200


class StationColors:
    """Color scheme for different station types."""
    COLORS = {
        StationType.CIRCLE: (255, 100, 100),      # Red
        StationType.TRIANGLE: (100, 255, 100),    # Green
        StationType.SQUARE: (100, 100, 255),      # Blue
        StationType.PENTAGON: (255, 255, 100),    # Yellow
        StationType.HEXAGON: (255, 100, 255),     # Magenta
        StationType.DIAMOND: (100, 255, 255),     # Cyan
        StationType.STAR: (255, 200, 100),        # Orange
        StationType.CROSS: (200, 100, 255),       # Purple
    }


class LineColors:
    """Color scheme for metro lines."""
    COLORS = {
        LineColor.RED: (220, 20, 20),
        LineColor.BLUE: (20, 20, 220),
        LineColor.GREEN: (20, 220, 20),
        LineColor.YELLOW: (220, 220, 20),
        LineColor.ORANGE: (255, 140, 0),
        LineColor.PURPLE: (160, 20, 220),
        LineColor.BROWN: (139, 69, 19),
        LineColor.PINK: (255, 20, 147),
        LineColor.GRAY: (128, 128, 128),
        LineColor.CYAN: (0, 255, 255),
    }


class AnimatedTrain:
    """Animated train representation for smooth movement."""
    
    def __init__(self, train_id: int, line_id: int, position: Tuple[float, float]):
        """
        Initialize animated train.
        
        Args:
            train_id: Train identifier
            line_id: Line identifier  
            position: Initial position
        """
        self.train_id = train_id
        self.line_id = line_id
        self.position = position
        self.target_position = position
        self.animation_progress = 0.0
        self.animation_speed = 2.0  # Pixels per frame
        
        # Visual properties
        self.passenger_count = 0
        self.capacity = 6
        self.is_at_station = False
        self.direction_angle = 0.0
        
        # Animation trail
        self.trail_positions = deque(maxlen=10)
        self.trail_positions.append(position)
    
    def update_target(self, new_position: Tuple[float, float]) -> None:
        """Update target position for animation."""
        if new_position != self.target_position:
            self.target_position = new_position
            self.animation_progress = 0.0
            
            # Calculate direction angle
            dx = new_position[0] - self.position[0]
            dy = new_position[1] - self.position[1]
            if dx != 0 or dy != 0:
                self.direction_angle = math.atan2(dy, dx)
    
    def update_animation(self, dt: float) -> None:
        """Update animation state."""
        if self.animation_progress < 1.0:
            # Animate towards target
            self.animation_progress = min(1.0, self.animation_progress + self.animation_speed * dt)
            
            # Interpolate position
            self.position = interpolate_positions(
                self.position,
                self.target_position,
                self.animation_progress
            )
            
            # Add to trail
            self.trail_positions.append(self.position)


class PygameRenderer:
    """
    Pygame-based renderer for Mini Metro visualization.
    
    Features:
    - Real-time game state rendering
    - Smooth train animations
    - Station and passenger visualization
    - Performance metrics display
    - Interactive controls
    """
    
    def __init__(self, config: RenderConfig = None):
        """
        Initialize pygame renderer.
        
        Args:
            config: Rendering configuration
        """
        self.config = config or RenderConfig()
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Mini Metro RL")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font = pygame.font.Font(None, self.config.font_size)
        self.small_font = pygame.font.Font(None, 12)
        
        # Animation state
        self.animated_trains: Dict[int, AnimatedTrain] = {}
        self.last_update_time = pygame.time.get_ticks()
        
        # Camera/viewport
        self.camera_offset = (0, 0)
        self.zoom_level = 1.0
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)
        
        logger.info(f"Initialized Pygame renderer: {self.config.width}x{self.config.height}")
    
    def render(self, game_state: Dict[str, Any]) -> None:
        """
        Render the current game state.
        
        Args:
            game_state: Current game state dictionary
        """
        current_time = pygame.time.get_ticks()
        dt = (current_time - self.last_update_time) / 1000.0
        self.last_update_time = current_time
        
        # Clear screen
        self.screen.fill(self.config.background_color)
        
        # Render game elements
        self._render_water_bodies(game_state.get('water_bodies', []))
        self._render_lines(game_state.get('lines', {}))
        self._render_stations(game_state.get('stations', {}))
        self._update_and_render_trains(game_state.get('lines', {}), dt)
        
        # Render UI
        if self.config.show_performance_metrics:
            self._render_performance_metrics(game_state.get('statistics', {}))
        
        self._render_game_info(game_state)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.config.fps)
        
        # Track frame time
        self.frame_times.append(dt)
    
    def _render_water_bodies(self, water_bodies: List[Tuple[Tuple[float, float], float]]) -> None:
        """Render water bodies (rivers, lakes)."""
        for (center_x, center_y), radius in water_bodies:
            pygame.draw.circle(
                self.screen,
                (100, 150, 200),  # Water blue
                (int(center_x), int(center_y)),
                int(radius),
                0
            )
    
    def _render_stations(self, stations: Dict[str, Any]) -> None:
        """Render all stations."""
        for station_id_str, station_data in stations.items():
            try:
                station_id = int(station_id_str)
                self._render_single_station(station_id, station_data)
            except (ValueError, TypeError):
                continue
    
    def _render_single_station(self, station_id: int, station_data: Any) -> None:
        """Render a single station."""
        # Extract station information from state vector or detailed state
        if isinstance(station_data, list):
            # State vector format - need to decode
            if len(station_data) >= 2:
                pos_x, pos_y = station_data[0] * self.config.width, station_data[1] * self.config.height
                # Try to determine station type from state vector
                station_type = StationType.CIRCLE  # Default
                queue_ratio = station_data[2] if len(station_data) > 2 else 0.0
            else:
                return
        elif isinstance(station_data, dict):
            # Detailed state format
            position = station_data.get('position', (0, 0))
            pos_x, pos_y = position
            station_type_str = station_data.get('station_type', 'circle')
            try:
                station_type = StationType(station_type_str)
            except ValueError:
                station_type = StationType.CIRCLE
            queue_ratio = station_data.get('queue_ratio', 0.0)
        else:
            return
        
        # Get station color
        station_color = StationColors.COLORS.get(station_type, (255, 255, 255))
        
        # Draw station
        pygame.draw.circle(
            self.screen,
            station_color,
            (int(pos_x), int(pos_y)),
            self.config.station_radius,
            0
        )
        
        # Draw station border
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (int(pos_x), int(pos_y)),
            self.config.station_radius,
            2
        )
        
        # Draw queue indicator if enabled
        if self.config.show_station_queues and queue_ratio > 0:
            queue_height = int(queue_ratio * 30)  # Max 30 pixels
            queue_rect = pygame.Rect(
                int(pos_x) + self.config.station_radius + 5,
                int(pos_y) - queue_height // 2,
                self.config.station_queue_width,
                queue_height
            )
            
            # Color based on queue level
            if queue_ratio > 0.8:
                queue_color = (255, 0, 0)  # Red - overloaded
            elif queue_ratio > 0.5:
                queue_color = (255, 255, 0)  # Yellow - getting full
            else:
                queue_color = (0, 255, 0)  # Green - normal
            
            pygame.draw.rect(self.screen, queue_color, queue_rect)
        
        # Draw station ID
        id_text = self.small_font.render(str(station_id), True, (255, 255, 255))
        self.screen.blit(id_text, (pos_x - 5, pos_y + self.config.station_radius + 2))
    
    def _render_lines(self, lines: Dict[str, Any]) -> None:
        """Render all metro lines."""
        for line_id_str, line_data in lines.items():
            try:
                line_id = int(line_id_str)
                self._render_single_line(line_id, line_data)
            except (ValueError, TypeError):
                continue
    
    def _render_single_line(self, line_id: int, line_data: Dict[str, Any]) -> None:
        """Render a single metro line."""
        stations = line_data.get('stations', [])
        if len(stations) < 2:
            return
        
        # Get line color
        line_color_str = line_data.get('color', 'red')
        try:
            line_color_enum = LineColor(line_color_str)
            line_color = LineColors.COLORS.get(line_color_enum, (255, 255, 255))
        except ValueError:
            line_color = (255, 255, 255)
        
        # Draw line segments between consecutive stations
        for i in range(len(stations) - 1):
            station1_id = stations[i]
            station2_id = stations[i + 1]
            
            # Get station positions (simplified - in real implementation would look up actual positions)
            pos1 = (100 + station1_id * 50, 200 + (station1_id % 3) * 100)
            pos2 = (100 + station2_id * 50, 200 + (station2_id % 3) * 100)
            
            pygame.draw.line(
                self.screen,
                line_color,
                pos1,
                pos2,
                self.config.line_width
            )
        
        # Draw loop closure if applicable
        if line_data.get('line_type') == 'loop' and len(stations) > 2:
            first_station = stations[0]
            last_station = stations[-1]
            
            pos1 = (100 + first_station * 50, 200 + (first_station % 3) * 100)
            pos2 = (100 + last_station * 50, 200 + (last_station % 3) * 100)
            
            pygame.draw.line(
                self.screen,
                line_color,
                pos1,
                pos2,
                self.config.line_width
            )
    
    def _update_and_render_trains(self, lines: Dict[str, Any], dt: float) -> None:
        """Update train animations and render trains."""
        current_trains = set()
        
        for line_id_str, line_data in lines.items():
            try:
                line_id = int(line_id_str)
            except (ValueError, TypeError):
                continue
            
            trains = line_data.get('trains', [])
            
            for train_data in trains:
                train_id = train_data.get('train_id', 0)
                current_trains.add(train_id)
                
                position = train_data.get('position', (0, 0))
                passenger_count = train_data.get('passenger_count', 0)
                capacity = train_data.get('capacity', 6)
                is_at_station = train_data.get('is_at_station', False)
                
                # Create or update animated train
                if train_id not in self.animated_trains:
                    self.animated_trains[train_id] = AnimatedTrain(train_id, line_id, position)
                
                animated_train = self.animated_trains[train_id]
                animated_train.update_target(position)
                animated_train.passenger_count = passenger_count
                animated_train.capacity = capacity
                animated_train.is_at_station = is_at_station
                animated_train.update_animation(dt)
                
                # Render train
                self._render_single_train(animated_train, line_data.get('color', 'red'))
        
        # Remove trains that no longer exist
        trains_to_remove = set(self.animated_trains.keys()) - current_trains
        for train_id in trains_to_remove:
            del self.animated_trains[train_id]
    
    def _render_single_train(self, train: AnimatedTrain, line_color: str) -> None:
        """Render a single train."""
        x, y = train.position
        
        # Get line color
        try:
            line_color_enum = LineColor(line_color)
            color = LineColors.COLORS.get(line_color_enum, (255, 255, 255))
        except ValueError:
            color = (255, 255, 255)
        
        # Draw train body
        train_rect = pygame.Rect(
            int(x) - self.config.train_size // 2,
            int(y) - self.config.train_size // 2,
            self.config.train_size,
            self.config.train_size
        )
        pygame.draw.rect(self.screen, color, train_rect, 0)
        pygame.draw.rect(self.screen, (255, 255, 255), train_rect, 1)
        
        # Draw direction indicator
        direction_length = 8
        end_x = x + direction_length * math.cos(train.direction_angle)
        end_y = y + direction_length * math.sin(train.direction_angle)
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (int(x), int(y)),
            (int(end_x), int(end_y)),
            2
        )
        
        # Draw passenger load indicator if enabled
        if self.config.show_train_loads:
            load_ratio = train.passenger_count / max(1, train.capacity)
            load_width = int(load_ratio * self.config.train_size)
            
            if load_width > 0:
                load_rect = pygame.Rect(
                    int(x) - self.config.train_size // 2,
                    int(y) - self.config.train_size // 2 - 3,
                    load_width,
                    2
                )
                
                # Color based on load
                if load_ratio > 0.8:
                    load_color = (255, 0, 0)  # Red - full
                elif load_ratio > 0.5:
                    load_color = (255, 255, 0)  # Yellow - getting full
                else:
                    load_color = (0, 255, 0)  # Green - light load
                
                pygame.draw.rect(self.screen, load_color, load_rect)
        
        # Draw trail
        if len(train.trail_positions) > 1:
            for i in range(len(train.trail_positions) - 1):
                alpha = i / len(train.trail_positions)
                trail_color = tuple(int(c * alpha * 0.5) for c in color)
                pygame.draw.line(
                    self.screen,
                    trail_color,
                    train.trail_positions[i],
                    train.trail_positions[i + 1],
                    1
                )
    
    def _render_performance_metrics(self, statistics: Dict[str, Any]) -> None:
        """Render performance metrics panel."""
        panel_rect = pygame.Rect(
            self.config.width - self.config.metric_panel_width,
            0,
            self.config.metric_panel_width,
            self.config.height
        )
        
        # Semi-transparent background
        panel_surface = pygame.Surface((self.config.metric_panel_width, self.config.height))
        panel_surface.set_alpha(200)
        panel_surface.fill((0, 0, 0))
        self.screen.blit(panel_surface, panel_rect.topleft)
        
        # Render metrics
        y_offset = 10
        line_height = 20
        
        metrics = [
            ("Passengers Delivered", statistics.get('passengers_delivered', 0)),
            ("Passengers Spawned", statistics.get('passengers_spawned', 0)),
            ("Delivery Rate", f"{statistics.get('delivery_rate', 0.0):.2f}"),
            ("Satisfaction", f"{statistics.get('average_satisfaction', 0.0):.2f}"),
            ("Game Duration", f"{statistics.get('game_duration', 0.0):.1f}s"),
            ("Max Passengers", statistics.get('max_simultaneous_passengers', 0)),
        ]
        
        for label, value in metrics:
            text = self.font.render(f"{label}: {value}", True, (255, 255, 255))
            self.screen.blit(text, (panel_rect.left + 10, y_offset))
            y_offset += line_height
        
        # Add FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / max(avg_frame_time, 0.001)
            fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (panel_rect.left + 10, y_offset))
    
    def _render_game_info(self, game_state: Dict[str, Any]) -> None:
        """Render basic game information."""
        current_week = game_state.get('current_week', 1)
        game_time = game_state.get('current_time', 0.0)
        game_state_str = game_state.get('game_state', 'unknown')
        
        # Game state text
        info_text = f"Week {current_week} | Time: {game_time:.1f}s | State: {game_state_str}"
        text_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        # Instructions
        instructions = [
            "ESC: Quit",
            "P: Pause/Resume",
            "R: Reset",
            "Space: Step (when paused)"
        ]
        
        for i, instruction in enumerate(instructions):
            text_surface = self.small_font.render(instruction, True, (200, 200, 200))
            self.screen.blit(text_surface, (10, self.config.height - 60 + i * 12))
    
    def handle_events(self) -> Dict[str, bool]:
        """
        Handle pygame events.
        
        Returns:
            Dictionary with event flags
        """
        events = {
            'quit': False,
            'pause': False,
            'reset': False,
            'step': False
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events['quit'] = True
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events['quit'] = True
                elif event.key == pygame.K_p:
                    events['pause'] = True
                elif event.key == pygame.K_r:
                    events['reset'] = True
                elif event.key == pygame.K_SPACE:
                    events['step'] = True
        
        return events
    
    def close(self) -> None:
        """Clean up pygame resources."""
        pygame.quit()
        logger.info("Pygame renderer closed")