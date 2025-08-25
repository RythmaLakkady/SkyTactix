import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import copy

from .simulation import SimulationEngine, SimulationConfig
from .physics import PhysicsConfig
from .spaces import SpaceManager

class AerialCombatEnv(gym.Env):    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10
    }
    
    def __init__(self, 
                num_agents: int = 4,
                team_config: Optional[Dict[int, str]] = None,
                sim_config: Optional[SimulationConfig] = None,
                physics_config: Optional[PhysicsConfig] = None,
                render_mode: Optional[str] = None):

        super().__init__()
        
        self.num_agents = num_agents
        self.render_mode = render_mode
        
        if team_config is None:
            team_config = {}
            for i in range(num_agents):
                team_config[i] = 'team_1' if i < num_agents // 2 else 'team_2'
        self.team_config = team_config
        
        if sim_config is None:
            sim_config = SimulationConfig()
            sim_config.max_agents = num_agents
        self.sim_config = sim_config
        self.physics_config = physics_config or PhysicsConfig()
        self.simulation = SimulationEngine(self.sim_config, self.physics_config)
        
        self.action_space = self.simulation.space_manager.get_action_space()
        self.observation_space = self.simulation.space_manager.get_observation_space()
        
        self.agent_ids = list(range(num_agents))
        self.possible_agents = self.agent_ids.copy()
        self.agents = self.agent_ids.copy()
        
        self.screen = None
        self.clock = None
        self.render_scale = 0.5
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[int, np.ndarray], Dict[int, Any]]:
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.agent_ids.copy()
        
        observations = self.simulation.reset(self.team_config)
        
        info = {agent_id: {} for agent_id in self.agents}
        
        return observations, info
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], 
                                                   Dict[int, bool], Dict[int, bool], Dict[int, Any]]:
        valid_actions = {aid: action for aid, action in actions.items() 
                        if aid in self.simulation.state.agents and self.simulation.state.agents[aid].alive}
        
        observations, rewards, terminated, truncated, info = self.simulation.step(valid_actions)
        
        self.agents = [aid for aid in self.agents 
                      if aid in self.simulation.state.agents and self.simulation.state.agents[aid].alive]
        
        return observations, rewards, terminated, truncated, info
    
    def render(self):
        if self.render_mode == 'human':
            return self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        
        try:
            import pygame
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((800, 600))
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.screen.fill((135, 206, 235))
        
        world_rect = pygame.Rect(
            50, 50, 
            int(self.physics_config.world_width * self.render_scale),
            int(self.physics_config.world_height * self.render_scale)
        )
        pygame.draw.rect(self.screen, (0, 100, 0), world_rect, 2)
        
        for agent_id, agent in self.simulation.state.agents.items():
            if not agent.alive:
                continue
                
            screen_x = 50 + int(agent.position[0] * self.render_scale)
            screen_y = 50 + int(agent.position[1] * self.render_scale)
            
            team = self.team_config.get(agent_id, 'team_1')
            if team == 'team_1':
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            
            radius = 5 + int(agent.altitude) * 2
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)
            
            heading_rad = np.radians(agent.heading)
            end_x = screen_x + int(15 * np.cos(heading_rad))
            end_y = screen_y + int(15 * np.sin(heading_rad))
            pygame.draw.line(self.screen, color, (screen_x, screen_y), (end_x, end_y), 2)
            
            health_ratio = agent.health / 100.0
            health_width = int(20 * health_ratio)
            health_rect = pygame.Rect(screen_x - 10, screen_y - 15, 20, 3)
            pygame.draw.rect(self.screen, (255, 0, 0), health_rect)
            if health_width > 0:
                health_fill = pygame.Rect(screen_x - 10, screen_y - 15, health_width, 3)
                pygame.draw.rect(self.screen, (0, 255, 0), health_fill)
        
        font = pygame.font.Font(None, 36)
        time_text = font.render(f"Time: {self.simulation.state.time:.1f}s", True, (0, 0, 0))
        self.screen.blit(time_text, (10, 10))
        
        step_text = font.render(f"Step: {self.simulation.state.step_count}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 50))
        
        if self.simulation.state.episode_done:
            winner_text = font.render(f"Winner: {self.simulation.state.winner}", True, (0, 0, 0))
            self.screen.blit(winner_text, (10, 90))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_rgb_array(self):
        width, height = 800, 600
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        rgb_array[:, :] = [135, 206, 235]
        
        return rgb_array
    
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
    
    def get_agent_observations(self) -> Dict[int, np.ndarray]:
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self.simulation.space_manager.build_observation(
                agent_id, self.simulation.state.agents, self.simulation.state.messages
            )
        return observations
    
    def get_simulation_info(self) -> Dict[str, Any]:
        return self.simulation.get_state_info()
    
    def get_action_meanings(self) -> List[str]:
        return self.simulation.space_manager.get_action_meanings()


class MultiAgentAerialCombatEnv:
    
    def __init__(self, **kwargs):
        self.env = AerialCombatEnv(**kwargs)
        self.possible_agents = self.env.possible_agents.copy()
        self.agents = self.env.agents.copy()
        
        self.action_spaces = {agent: self.env.action_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self.env.observation_space for agent in self.possible_agents}
    
    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.agents = self.env.agents.copy()
        return observations, infos
    
    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self.agents = self.env.agents.copy()
        return observations, rewards, terminated, truncated, infos
    
    def render(self, mode='human'):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    @property
    def unwrapped(self):
        return self.env
