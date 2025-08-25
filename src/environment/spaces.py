#state and action space definitions

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
from enum import IntEnum

from .physics import AltitudeLayer, PhysicsConfig

class Action(IntEnum):
    TURN_LEFT = 0      
    TURN_RIGHT = 1     
    SPEED_UP = 2       
    SPEED_DOWN = 3     
    CLIMB = 4          
    DESCEND = 5        
    FIRE_WEAPON = 6    
    SEND_MESSAGE = 7   

class AgentState:
    
    def __init__(self, position=None, heading=0.0, speed=20.0, altitude=AltitudeLayer.MEDIUM,
                health=100.0, fuel=1000.0, alive=True, last_fire_time=0.0, 
                weapon_ready=True, last_message=None):
        
        self.position = position if position is not None else np.array([0.0, 0.0])
        self.heading = heading            
        self.speed = speed                 
        self.altitude = altitude           
        
        self.health = health               
        self.fuel = fuel                   
        self.alive = alive                 
        
        self.last_fire_time = last_fire_time  
        self.weapon_ready = weapon_ready      
        
        self.last_message = last_message if last_message is not None else np.zeros(4)
    
    def to_observation_vector(self, normalize=True):
        obs = np.array([
            self.position[0],
            self.position[1], 
            self.heading,
            self.speed,
            float(self.altitude),
            self.health,
            self.fuel,
            float(self.alive),
            float(self.weapon_ready)
        ])
        
        if normalize:
            obs[0] /= 500.0 
            obs[1] /= 500.0  
            obs[2] /= 180.0 
            obs[3] /= 100.0 
            obs[4] /= 2.0    
            obs[5] /= 100.0 
            obs[6] /= 1000.0 
            
        return obs

class SpaceManager:
    
    def __init__(self, config: PhysicsConfig, max_agents: int = 8, 
                message_size: int = 4, communication_enabled: bool = False):
        self.config = config
        self.max_agents = max_agents
        self.message_size = message_size
        self.communication_enabled = communication_enabled
        
        self.turn_amount = 15.0      
        self.speed_change = 10.0     
        
    def get_action_space(self) -> spaces.Discrete:
        num_actions = len(Action)
        return spaces.Discrete(num_actions)
    
    def get_observation_space(self) -> spaces.Box:
        own_state_dim = 9
        other_agents_dim = (self.max_agents - 1) * 6
        
        comm_dim = self.message_size if self.communication_enabled else 0
        
        total_dim = own_state_dim + other_agents_dim + comm_dim
        
        return spaces.Box(
            low=-2.0,
            high=2.0, 
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def action_to_commands(self, action: int, current_state: AgentState) -> Dict[str, Any]:
        commands = {
            'turn_command': 0.0,
            'speed_command': current_state.speed,
            'altitude_command': current_state.altitude,
            'fire_weapon': False,
            'send_message': False,
            'message_content': None
        }
        
        if action == Action.TURN_LEFT:
            commands['turn_command'] = -self.turn_amount
        elif action == Action.TURN_RIGHT:
            commands['turn_command'] = self.turn_amount
        elif action == Action.SPEED_UP:
            commands['speed_command'] = min(
                current_state.speed + self.speed_change, 
                self.config.max_speed
            )
        elif action == Action.SPEED_DOWN:
            commands['speed_command'] = max(
                current_state.speed - self.speed_change,
                self.config.min_speed
            )
        elif action == Action.CLIMB:
            if current_state.altitude < AltitudeLayer.HIGH:
                commands['altitude_command'] = AltitudeLayer(current_state.altitude + 1)
        elif action == Action.DESCEND:
            if current_state.altitude > AltitudeLayer.LOW:
                commands['altitude_command'] = AltitudeLayer(current_state.altitude - 1)
        elif action == Action.FIRE_WEAPON:
            commands['fire_weapon'] = True
        elif action == Action.SEND_MESSAGE:
            commands['send_message'] = True
            commands['message_content'] = np.random.rand(self.message_size)
            
        return commands
    
    def build_observation(self, agent_id: int, agent_states: Dict[int, AgentState], 
                         messages: Dict[int, np.ndarray] = None) -> np.ndarray:
        if agent_id not in agent_states:
            return np.zeros(self.get_observation_space().shape[0])
            
        agent = agent_states[agent_id]
        obs_parts = []
        
        own_obs = agent.to_observation_vector(normalize=True)
        obs_parts.append(own_obs)
        
        other_agents = [aid for aid in agent_states.keys() if aid != agent_id and agent_states[aid].alive]
        
        for i in range(self.max_agents - 1):
            if i < len(other_agents):
                other_id = other_agents[i]
                other_agent = agent_states[other_id]
                
                rel_pos = other_agent.position - agent.position
                rel_pos = rel_pos / 500.0 
                
                rel_heading = (other_agent.heading - agent.heading) / 180.0
                
                other_obs = np.array([
                    rel_pos[0],
                    rel_pos[1],
                    rel_heading,
                    other_agent.speed / 100.0,
                    float(other_agent.altitude) / 2.0,
                    other_agent.health / 100.0
                ])
            else:
                other_obs = np.zeros(6)
                
            obs_parts.append(other_obs)
        
        if self.communication_enabled:
            if messages and agent_id in messages:
                msg_obs = messages[agent_id]
            else:
                msg_obs = np.zeros(self.message_size)
            obs_parts.append(msg_obs)
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def get_action_meanings(self) -> List[str]:
        return [
            "TURN_LEFT",
            "TURN_RIGHT", 
            "SPEED_UP",
            "SPEED_DOWN",
            "CLIMB",
            "DESCEND",
            "FIRE_WEAPON",
            "SEND_MESSAGE"
        ]
