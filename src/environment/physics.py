import numpy as np
from enum import IntEnum
from typing import Tuple, Dict, Any

class AltitudeLayer(IntEnum):
    LOW = 0
    MEDIUM = 1  
    HIGH = 2

class PhysicsConfig:
    def __init__(self):
        self.world_width = 1000.0
        self.world_height = 1000.0
        
        self.max_speed = 100.0  
        self.min_speed = 20.0  
        self.max_turn_rate = 90.0 
        
        self.max_fuel = 1000.0
        self.fuel_consumption_base = 1.0 
        self.fuel_consumption_afterburner = 3.0 
        
        self.altitude_agility_multiplier = {
            AltitudeLayer.LOW: 1.2,   
            AltitudeLayer.MEDIUM: 1.0,
            AltitudeLayer.HIGH: 0.8   
        }

        self.altitude_range_multiplier = {
            AltitudeLayer.LOW: 0.8,   
            AltitudeLayer.MEDIUM: 1.0,
            AltitudeLayer.HIGH: 1.3   
        }
            
        self.altitude_sensor_multiplier = {
            AltitudeLayer.LOW: 0.9,   
            AltitudeLayer.MEDIUM: 1.0,
            AltitudeLayer.HIGH: 1.4   
        }
        
        self.weapon_range_base = 150.0
        self.weapon_damage = 100.0
        self.weapon_cooldown = 2.0 

class PhysicsEngine:
    
    def __init__(self, config: PhysicsConfig = None):
        self.config = config or PhysicsConfig()
    
    def update_position(self, position: np.ndarray, heading: float, speed: float, dt: float) -> np.ndarray:
        
        heading_rad = np.radians(heading)
        
        vx = speed * np.cos(heading_rad)
        vy = speed * np.sin(heading_rad)
        
        new_position = position + np.array([vx, vy]) * dt
        
        new_position[0] = new_position[0] % self.config.world_width
        new_position[1] = new_position[1] % self.config.world_height
        
        return new_position
    
    def update_heading(self, current_heading: float, turn_command: float, altitude: AltitudeLayer, dt: float) -> float:
        agility_multiplier = self.config.altitude_agility_multiplier[altitude]
        max_turn = self.config.max_turn_rate * agility_multiplier * dt
        
        actual_turn = np.clip(turn_command, -max_turn, max_turn)
        
        new_heading = (current_heading + actual_turn) % 360.0
        
        return new_heading
    
    def calculate_fuel_consumption(self, speed: float, dt: float) -> float:
        speed_ratio = (speed - self.config.min_speed) / (self.config.max_speed - self.config.min_speed)
        speed_ratio = np.clip(speed_ratio, 0.0, 1.0)
        
        consumption_rate = (
            self.config.fuel_consumption_base + 
            speed_ratio * (self.config.fuel_consumption_afterburner - self.config.fuel_consumption_base)
        )
        
        return consumption_rate * dt
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)
    
    def calculate_relative_bearing(self, from_pos: np.ndarray, to_pos: np.ndarray, from_heading: float) -> float:
        delta = to_pos - from_pos
        absolute_bearing = np.degrees(np.arctan2(delta[1], delta[0]))
        
        relative_bearing = (absolute_bearing - from_heading) % 360.0
        
        if relative_bearing > 180.0:
            relative_bearing -= 360.0
            
        return relative_bearing
    
    def can_fire_weapon(self, shooter_pos: np.ndarray, target_pos: np.ndarray, 
                    shooter_altitude: AltitudeLayer, last_fire_time: float, current_time: float) -> bool:
            
        if current_time - last_fire_time < self.config.weapon_cooldown:
            return False
        
        distance = self.calculate_distance(shooter_pos, target_pos)
        max_range = self.config.weapon_range_base * self.config.altitude_range_multiplier[shooter_altitude]
        
        return distance <= max_range
    
    def calculate_hit_probability(self, shooter_pos: np.ndarray, target_pos: np.ndarray,
                                shooter_heading: float, target_heading: float,
                                shooter_altitude: AltitudeLayer) -> float:
        distance = self.calculate_distance(shooter_pos, target_pos)
        max_range = self.config.weapon_range_base * self.config.altitude_range_multiplier[shooter_altitude]
        
        if distance > max_range:
            return 0.0
        
        distance_factor = 1.0 - (distance / max_range)
        
        relative_bearing = abs(self.calculate_relative_bearing(shooter_pos, target_pos, shooter_heading))
        angle_factor = max(0.0, 1.0 - relative_bearing / 90.0)  # Maximum accuracy when directly ahead
        
        altitude_factor = self.config.altitude_range_multiplier[shooter_altitude]
        
        hit_probability = distance_factor * angle_factor * altitude_factor * 0.8  # Base 80% max hit rate
        
        return np.clip(hit_probability, 0.0, 1.0)
