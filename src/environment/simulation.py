import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time

from .physics import PhysicsEngine, PhysicsConfig, AltitudeLayer
from .spaces import SpaceManager, AgentState, Action

class SimulationConfig:
    
    def __init__(self):
        self.max_agents = 8
        self.max_episode_steps = 1000
        self.dt = 0.1  
        
        self.elimination_victory = True  
        self.time_limit_victory = True   
        
        self.spawn_radius = 100.0       
        self.min_spawn_distance = 50.0  
        
        self.enable_communication = False
        self.message_size = 4

class SimulationState:
    
    def __init__(self):
        self.agents = {}  
        self.time = 0.0
        self.step_count = 0
        self.episode_done = False
        self.winner = None  
        
        self.messages = {}  
    
class SimulationEngine:
    
    def __init__(self, sim_config: SimulationConfig = None, physics_config: PhysicsConfig = None):
        self.sim_config = sim_config or SimulationConfig()
        self.physics_config = physics_config or PhysicsConfig()
        
        self.physics_engine = PhysicsEngine(self.physics_config)
        self.space_manager = SpaceManager(
            self.physics_config, 
            self.sim_config.max_agents,
            self.sim_config.message_size,
            self.sim_config.enable_communication
        )
        
        self.state = SimulationState()
        self.team_assignments = {}  
        
    def reset(self, agent_teams: Dict[int, str] = None) -> Dict[int, np.ndarray]:
        self.state = SimulationState()
        self.team_assignments = agent_teams or {}
        
        if agent_teams:
            self._spawn_agents(list(agent_teams.keys()))
        
        observations = {}
        for agent_id in self.state.agents:
            observations[agent_id] = self.space_manager.build_observation(
                agent_id, self.state.agents, self.state.messages
            )
            
        return observations
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], 
                                                Dict[int, bool], Dict[int, bool], Dict[int, Any]]:
        commands = {}
        for agent_id, action in actions.items():
            if agent_id in self.state.agents and self.state.agents[agent_id].alive:
                commands[agent_id] = self.space_manager.action_to_commands(
                    action, self.state.agents[agent_id]
                )
        
        self._update_agents(commands)
        
        self._handle_combat()
        
        self.state.time += self.sim_config.dt
        self.state.step_count += 1
        
        self._check_termination_conditions()
        
        rewards = self._calculate_rewards()
        
        observations = {}
        terminated = {}
        truncated = {}
        
        for agent_id in self.state.agents:
            observations[agent_id] = self.space_manager.build_observation(
                agent_id, self.state.agents, self.state.messages
            )
            terminated[agent_id] = not self.state.agents[agent_id].alive or self.state.episode_done
            truncated[agent_id] = self.state.step_count >= self.sim_config.max_episode_steps
        
        info = {
            'simulation_time': self.state.time,
            'step_count': self.state.step_count,
            'winner': self.state.winner,
            'agents_alive': {aid: agent.alive for aid, agent in self.state.agents.items()}
        }
        
        return observations, rewards, terminated, truncated, {agent_id: info for agent_id in self.state.agents}
    
    def _spawn_agents(self, agent_ids: List[int]):
        spawn_positions = self._generate_spawn_positions(len(agent_ids))
        
        for i, agent_id in enumerate(agent_ids):
            
            position = spawn_positions[i]
            heading = np.random.uniform(0, 360)
            speed = self.physics_config.min_speed + np.random.uniform(0, 20)
            altitude = AltitudeLayer(np.random.randint(0, 3))
            

            self.state.agents[agent_id] = AgentState(
                position=position,
                heading=heading,
                speed=speed,
                altitude=altitude,
                health=100.0,
                fuel=self.physics_config.max_fuel,
                alive=True,
                last_fire_time=0.0,
                weapon_ready=True,
                last_message=np.zeros(self.sim_config.message_size)
            )
    
    def _generate_spawn_positions(self, num_agents: int) -> List[np.ndarray]:
        positions = []
        max_attempts = 100
        
        for _ in range(num_agents):
            for attempt in range(max_attempts):
                center = np.array([
                    self.physics_config.world_width / 2,
                    self.physics_config.world_height / 2
                ])
                
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, self.sim_config.spawn_radius)
                
                position = center + radius * np.array([np.cos(angle), np.sin(angle)])
                
                if not positions or all(
                    np.linalg.norm(position - pos) >= self.sim_config.min_spawn_distance 
                    for pos in positions
                ):
                    positions.append(position)
                    break
            else:
                positions.append(center + np.random.uniform(-50, 50, 2))
        
        return positions
    
    def _update_agents(self, commands: Dict[int, Dict[str, Any]]):
        dt = self.sim_config.dt
        
        for agent_id, agent in self.state.agents.items():
            if not agent.alive:
                continue
                
            if agent_id not in commands:
                continue
                
            cmd = commands[agent_id]
            
            agent.position = self.physics_engine.update_position(
                agent.position, agent.heading, agent.speed, dt
            )
            
            agent.heading = self.physics_engine.update_heading(
                agent.heading, cmd['turn_command'], agent.altitude, dt
            )
            
            agent.speed = cmd['speed_command']
            
            agent.altitude = cmd['altitude_command']
            
            fuel_consumed = self.physics_engine.calculate_fuel_consumption(agent.speed, dt)
            agent.fuel = max(0.0, agent.fuel - fuel_consumed)
            
            if agent.fuel <= 0:
                agent.alive = False
                agent.health = 0.0
            
            time_since_fire = self.state.time - agent.last_fire_time
            agent.weapon_ready = time_since_fire >= self.physics_config.weapon_cooldown
            
            if cmd.get('send_message', False) and self.sim_config.enable_communication:
                if cmd['message_content'] is not None:
                    self._broadcast_message(agent_id, cmd['message_content'])
    
    def _handle_combat(self):
        for agent_id, agent in self.state.agents.items():
            if not agent.alive:
                continue
            
            targets = self._find_targets_in_range(agent_id)
            
            if targets and agent.weapon_ready:
                closest_target = min(targets, key=lambda t: self.physics_engine.calculate_distance(
                    agent.position, self.state.agents[t].position
                ))
                
                hit_prob = self.physics_engine.calculate_hit_probability(
                    agent.position, self.state.agents[closest_target].position,
                    agent.heading, self.state.agents[closest_target].heading,
                    agent.altitude
                )
                
                if np.random.random() < hit_prob:
                    self._apply_damage(closest_target, self.physics_config.weapon_damage)
                    agent.last_fire_time = self.state.time
    
    def _find_targets_in_range(self, agent_id: int) -> List[int]:
        agent = self.state.agents[agent_id]
        targets = []
        
        agent_team = self.team_assignments.get(agent_id, "team_1")
        
        for other_id, other_agent in self.state.agents.items():
            if other_id == agent_id or not other_agent.alive:
                continue
                
            other_team = self.team_assignments.get(other_id, "team_2")
            
            if other_team != agent_team:
                if self.physics_engine.can_fire_weapon(
                    agent.position, other_agent.position, 
                    agent.altitude, agent.last_fire_time, self.state.time
                ):
                    targets.append(other_id)
        
        return targets
    
    def _apply_damage(self, target_id: int, damage: float):
        if target_id in self.state.agents:
            agent = self.state.agents[target_id]
            agent.health = max(0.0, agent.health - damage)
            
            if agent.health <= 0:
                agent.alive = False
    
    def _broadcast_message(self, sender_id: int, message: np.ndarray):
        sender_team = self.team_assignments.get(sender_id, "team_1")
        
        for agent_id in self.state.agents:
            if agent_id != sender_id and self.team_assignments.get(agent_id, "team_2") == sender_team:
                self.state.messages[agent_id] = message.copy()
    
    def _check_termination_conditions(self):
        if self.sim_config.elimination_victory:
            teams_alive = set()
            for agent_id, agent in self.state.agents.items():
                if agent.alive:
                    team = self.team_assignments.get(agent_id, "team_1")
                    teams_alive.add(team)
            
            if len(teams_alive) <= 1:
                self.state.episode_done = True
                self.state.winner = list(teams_alive)[0] if teams_alive else "draw"
        
        if self.sim_config.time_limit_victory:
            if self.state.step_count >= self.sim_config.max_episode_steps:
                self.state.episode_done = True
                if not self.state.winner:
                    self.state.winner = "draw"
    
    def _calculate_rewards(self) -> Dict[int, float]:
        rewards = {}
        
        for agent_id, agent in self.state.agents.items():
            reward = 0.0
            
            if agent.alive:
                reward += 0.1
            
            if agent.health < 100.0:
                reward -= (100.0 - agent.health) * 0.01
            
            fuel_ratio = agent.fuel / self.physics_config.max_fuel
            reward += fuel_ratio * 0.05
            
            if self.state.episode_done:
                agent_team = self.team_assignments.get(agent_id, "team_1")
                if self.state.winner == agent_team:
                    reward += 10.0
                elif self.state.winner == "draw":
                    reward += 2.0
                else:
                    reward -= 5.0
            
            rewards[agent_id] = reward
        
        return rewards
    
    def get_state_info(self) -> Dict[str, Any]:
        return {
            'time': self.state.time,
            'step_count': self.state.step_count,
            'episode_done': self.state.episode_done,
            'winner': self.state.winner,
            'agents': {aid: {
                'position': agent.position.tolist(),
                'heading': agent.heading,
                'speed': agent.speed,
                'altitude': int(agent.altitude),
                'health': agent.health,
                'fuel': agent.fuel,
                'alive': agent.alive
            } for aid, agent in self.state.agents.items()}
        }
