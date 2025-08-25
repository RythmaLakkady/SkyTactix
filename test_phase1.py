"""
Test script for Phase 1: Verify the 2.5D aerial combat environment works correctly.
This script tests the basic functionality without RL training.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.aerial_combat_env import AerialCombatEnv, MultiAgentAerialCombatEnv
from src.environment.physics import AltitudeLayer
from src.environment.spaces import Action

def test_basic_environment():
    """Test basic environment functionality"""
    print("=== Testing Basic Environment ===")
    
    # Create environment with 4 agents (2v2)
    team_config = {0: 'team_1', 1: 'team_1', 2: 'team_2', 3: 'team_2'}
    env = AerialCombatEnv(num_agents=4, team_config=team_config)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action meanings: {env.get_action_meanings()}")
    
    # Reset environment
    observations, info = env.reset(seed=42)
    print(f"Initial observations keys: {list(observations.keys())}")
    print(f"Observation shape: {observations[0].shape}")
    
    # Get simulation info
    sim_info = env.get_simulation_info()
    print(f"Initial simulation state:")
    for agent_id, agent_info in sim_info['agents'].items():
        print(f"  Agent {agent_id}: pos={agent_info['position']}, "
              f"heading={agent_info['heading']:.1f}°, "
              f"altitude={agent_info['altitude']}, "
              f"health={agent_info['health']}")
    
    return env

def test_agent_actions():
    """Test different agent actions"""
    print("\n=== Testing Agent Actions ===")
    
    team_config = {0: 'team_1', 1: 'team_2'}  # Simple 1v1
    env = AerialCombatEnv(num_agents=2, team_config=team_config)
    
    observations, _ = env.reset(seed=42)
    
    # Test various actions
    actions_to_test = [
        {0: Action.TURN_LEFT, 1: Action.TURN_RIGHT},
        {0: Action.SPEED_UP, 1: Action.SPEED_DOWN},
        {0: Action.CLIMB, 1: Action.DESCEND},
        {0: Action.FIRE_WEAPON, 1: Action.FIRE_WEAPON}
    ]
    
    for step, actions in enumerate(actions_to_test):
        print(f"\nStep {step + 1}: Testing actions {actions}")
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        sim_info = env.get_simulation_info()
        for agent_id in [0, 1]:
            if agent_id in sim_info['agents']:
                agent = sim_info['agents'][agent_id]
                print(f"  Agent {agent_id}: pos=({agent['position'][0]:.1f}, {agent['position'][1]:.1f}), "
                      f"heading={agent['heading']:.1f}°, speed={agent['speed']:.1f}, "
                      f"altitude={agent['altitude']}, health={agent['health']:.1f}")
        
        print(f"  Rewards: {rewards}")
        print(f"  Terminated: {terminated}")
        
        if any(terminated.values()):
            print("Episode terminated!")
            break
    
    return env

def test_combat_scenario():
    """Test a simple combat scenario"""
    print("\n=== Testing Combat Scenario ===")
    
    team_config = {0: 'team_1', 1: 'team_2'}
    env = AerialCombatEnv(num_agents=2, team_config=team_config)
    
    observations, _ = env.reset(seed=123)  # Different seed for variety
    
    max_steps = 50
    step = 0
    
    while step < max_steps:
        # Simple AI: agents move toward each other and fire
        actions = {}
        sim_info = env.get_simulation_info()
        
        for agent_id in env.agents:
            if agent_id not in sim_info['agents']:
                continue
                
            agent = sim_info['agents'][agent_id]
            
            # Find closest enemy
            closest_enemy = None
            min_distance = float('inf')
            
            for other_id, other_agent in sim_info['agents'].items():
                if other_id != agent_id and other_agent['alive']:
                    other_team = env.team_config.get(other_id, 'team_2')
                    agent_team = env.team_config.get(agent_id, 'team_1')
                    
                    if other_team != agent_team:
                        distance = np.sqrt(
                            (agent['position'][0] - other_agent['position'][0])**2 +
                            (agent['position'][1] - other_agent['position'][1])**2
                        )
                        if distance < min_distance:
                            min_distance = distance
                            closest_enemy = other_agent
            
            # Choose action based on simple strategy
            if closest_enemy:
                # Calculate bearing to enemy
                dx = closest_enemy['position'][0] - agent['position'][0]
                dy = closest_enemy['position'][1] - agent['position'][1]
                target_heading = np.degrees(np.arctan2(dy, dx)) % 360
                
                # Calculate heading difference
                heading_diff = (target_heading - agent['heading'] + 180) % 360 - 180
                
                if abs(heading_diff) > 15:
                    # Turn toward enemy
                    actions[agent_id] = Action.TURN_LEFT if heading_diff < 0 else Action.TURN_RIGHT
                elif min_distance > 100:
                    # Speed up to close distance
                    actions[agent_id] = Action.SPEED_UP
                else:
                    # Fire weapon
                    actions[agent_id] = Action.FIRE_WEAPON
            else:
                # No enemy found, just move forward
                actions[agent_id] = Action.SPEED_UP
        
        # Execute step
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 10 == 0:  # Print every 10 steps
            print(f"\nStep {step}:")
            sim_info = env.get_simulation_info()
            for agent_id in env.agents:
                if agent_id in sim_info['agents']:
                    agent = sim_info['agents'][agent_id]
                    team = env.team_config.get(agent_id, 'unknown')
                    print(f"  Agent {agent_id} ({team}): "
                          f"health={agent['health']:.1f}, fuel={agent['fuel']:.1f}")
        
        # Check if episode is done
        if any(terminated.values()) or any(truncated.values()):
            print(f"\nEpisode finished at step {step}!")
            print(f"Winner: {sim_info.get('winner', 'unknown')}")
            break
        
        step += 1
    
    if step >= max_steps:
        print(f"\nEpisode reached maximum steps ({max_steps})")
    
    return env

def test_multi_agent_wrapper():
    """Test the multi-agent wrapper"""
    print("\n=== Testing Multi-Agent Wrapper ===")
    
    team_config = {0: 'team_1', 1: 'team_1', 2: 'team_2', 3: 'team_2'}
    env = MultiAgentAerialCombatEnv(num_agents=4, team_config=team_config)
    
    print(f"Possible agents: {env.possible_agents}")
    print(f"Action spaces: {list(env.action_spaces.keys())}")
    print(f"Observation spaces: {list(env.observation_spaces.keys())}")
    
    observations, info = env.reset(seed=42)
    print(f"Active agents after reset: {env.agents}")
    
    # Test a few random steps
    for step in range(5):
        actions = {agent_id: np.random.randint(0, len(Action)) for agent_id in env.agents}
        observations, rewards, terminated, truncated, info = env.step(actions)
        print(f"Step {step + 1}: {len(env.agents)} agents active")
        
        if not env.agents:  # All agents eliminated
            print("All agents eliminated!")
            break
    
    env.close()
    return True

def main():
    """Run all tests"""
    print("Starting Phase 1 Environment Tests")
    print("=" * 50)
    
    try:
        # Test 1: Basic environment
        env1 = test_basic_environment()
        env1.close()
        
        # Test 2: Agent actions
        env2 = test_agent_actions()
        env2.close()
        
        # Test 3: Combat scenario
        env3 = test_combat_scenario()
        env3.close()
        
        # Test 4: Multi-agent wrapper
        test_multi_agent_wrapper()
        
        print("\n" + "=" * 50)
        print("✅ All Phase 1 tests completed successfully!")
        print("🎯 Environment is ready for RL training (Phase 2)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
