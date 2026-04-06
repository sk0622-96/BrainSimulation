"""
Survival Grid MDP - Can the organism reach the goal?

Setup:
- Start: Top-left corner (0, 0)
- Goal: Bottom-right corner (49, 49)
- Objective: Survive the journey and reach the goal!

The organism has HEALTH that changes as it encounters agents:
- Food: +20 health (survival resources)
- Threats: -30 health (dangerous traps)
- Info: +5 health (helpful information/rest stops)

Win condition: Reach goal with health > 0
Lose condition: Health drops to 0 or below
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# ==================== CONFIGURATION ====================

# Grid size
GRID_SIZE = 50

# Starting position (top-left)
START_POSITION = (0, 0)

# Goal position (bottom-right)
GOAL_POSITION = (49, 49)

# Agent effects on health
FOOD_HEALTH = 20        # Food restores health
THREAT_HEALTH = -30     # Threats damage health
INFO_HEALTH = 5         # Info gives small health boost

# Rewards
GOAL_REWARD = 100.0     # Big reward for reaching goal
DEATH_PENALTY = -50.0   # Penalty for dying
MOVE_COST = -0.1        # Small cost per move

# Starting health
STARTING_HEALTH = 100

# Actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']


# ==================== AGENT CLASS ====================

class Agent:
    """
    An agent in the environment (food, threat, info)
    """
    def __init__(self, x, y, agent_type, health_change):
        self.x = x                          # X position
        self.y = y                          # Y position
        self.agent_type = agent_type        # 'food', 'threat', or 'info'
        self.health_change = health_change  # How it affects organism health

# ==================== MDP CLASS ====================

class SurvivalMDP:
    """
    Markov Decision Process for survival navigation
    
    State: (x, y, health)
    Goal: Reach (49, 49) with health > 0
    """
    
    def __init__(self, grid_size=50, num_food=15, num_threats=10, num_info=8):
        """
        Initialize the survival MDP
        
        Args:
            grid_size: Size of square grid
            num_food: Number of food agents (help survival)
            num_threats: Number of threat agents (dangerous!)
            num_info: Number of info agents (small help)
        """
        self.grid_size = grid_size
        self.start = START_POSITION
        self.goal = GOAL_POSITION
        self.agents = []
        self.initial_agents = []  # For resetting
        
        # Generate random agents (avoid start and goal positions)
        np.random.seed(42)  # For reproducibility - change or remove for true randomness
        
        # Add food agents
        for _ in range(num_food):
            while True:
                x = np.random.randint(0, grid_size)
                y = np.random.randint(0, grid_size)
                # Don't place on start or goal
                if (x, y) != self.start and (x, y) != self.goal:
                    self.agents.append(Agent(x, y, 'food', FOOD_HEALTH))
                    break
        
        # Add threat agents
        for _ in range(num_threats):
            while True:
                x = np.random.randint(0, grid_size)
                y = np.random.randint(0, grid_size)
                if (x, y) != self.start and (x, y) != self.goal:
                    self.agents.append(Agent(x, y, 'threat', THREAT_HEALTH))
                    break
        
        # Add info agents
        for _ in range(num_info):
            while True:
                x = np.random.randint(0, grid_size)
                y = np.random.randint(0, grid_size)
                if (x, y) != self.start and (x, y) != self.goal:
                    self.agents.append(Agent(x, y, 'info', INFO_HEALTH))
                    break
        
        # Save initial configuration for reset
        self.initial_agents = [Agent(a.x, a.y, a.agent_type, a.health_change) 
                              for a in self.agents]
        
        print(f"\n{'='*60}")
        print(f"SURVIVAL MDP: {grid_size}x{grid_size} Grid")
        print(f"{'='*60}")
        print(f"Start: {self.start}")
        print(f"Goal:  {self.goal}")
        print(f"Starting Health: {STARTING_HEALTH}")
        print(f"\nAgents in environment:")
        print(f"  Food (health +{FOOD_HEALTH}):     {num_food}")
        print(f"  Threats (health {THREAT_HEALTH}): {num_threats}")
        print(f"  Info (health +{INFO_HEALTH}):      {num_info}")
        print(f"{'='*60}\n")
    
    def reset(self):
        """Reset environment to initial state"""
        self.agents = [Agent(a.x, a.y, a.agent_type, a.health_change) 
                      for a in self.initial_agents]
    
    def get_agent_at(self, x, y):
        """Check if there's an agent at position (x, y)"""
        for agent in self.agents:
            if agent.x == x and agent.y == y:
                return agent
        return None
    
    def is_valid_position(self, x, y):
        """Check if position is within grid bounds"""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def is_goal(self, x, y):
        """Check if position is the goal"""
        return (x, y) == self.goal
    
    def transition(self, state, action):
        """
        TRANSITION FUNCTION: P(s'|s,a)
        
        Args:
            state: (x, y, health) tuple
            action: One of ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        Returns:
            next_state: (new_x, new_y, new_health)
        """
        x, y, health = state
        
        # Determine new position based on action
        if action == 'UP':
            new_x, new_y = x, y - 1
        elif action == 'DOWN':
            new_x, new_y = x, y + 1
        elif action == 'LEFT':
            new_x, new_y = x - 1, y
        elif action == 'RIGHT':
            new_x, new_y = x + 1, y
        else:
            new_x, new_y = x, y
        
        # Check bounds - if we hit wall, stay in place
        if not self.is_valid_position(new_x, new_y):
            new_x, new_y = x, y
        
        # Health stays same for now (will be updated by reward function)
        new_health = health
        
        return (new_x, new_y, new_health)
    
    def get_reward(self, state, action, next_state):
        """
        REWARD FUNCTION: R(s,a,s')
        
        Returns reward AND updates health
        
        Args:
            state: Current (x, y, health)
            action: Action taken
            next_state: Next (x, y, health)
        
        Returns:
            (reward, new_health, status)
            status: 'continue', 'goal', or 'dead'
        """
        x, y, health = next_state
        reward = MOVE_COST  # Base cost for moving
        new_health = health
        status = 'continue'
        
        # Check if we reached the goal!
        if self.is_goal(x, y):
            reward += GOAL_REWARD
            status = 'goal'
            print(f"    GOAL REACHED! Health remaining: {new_health}")
            return reward, new_health, status
        
        # Check for agent encounter
        agent = self.get_agent_at(x, y)
        if agent is not None:
            # Update health based on agent type
            new_health += agent.health_change
            
            # Reward is proportional to health change
            reward += agent.health_change * 0.5
            
            # Print encounter
            if agent.agent_type == 'food':
                print(f"    Found food! Health: {health} -> {new_health}")
            elif agent.agent_type == 'threat':
                print(f"    Hit threat! Health: {health} -> {new_health}")
            else:
                print(f"    Found info! Health: {health} -> {new_health}")
            
            # Remove agent (it's been encountered)
            self.agents.remove(agent)
        
        # Check if organism died
        if new_health <= 0:
            reward += DEATH_PENALTY
            new_health = 0
            status = 'dead'
            print(f"    ORGANISM DIED!")
        
        return reward, new_health, status
    
    def simple_goal_policy(self, state):
        """
        POLICY: π(s) -> a
        
        Simple policy: Move toward goal (greedy Manhattan distance)
        NOTE: This is a "dumb" policy - just goes straight to goal
        Future: Make this smarter to avoid threats and seek food!
        
        Args:
            state: (x, y, health)
        
        Returns:
            action: Best action to take
        """
        x, y, health = state
        goal_x, goal_y = self.goal
        
        # Move toward goal (prioritize larger distance)
        dx = goal_x - x
        dy = goal_y - y
        
        if abs(dx) > abs(dy):
            # Move horizontally
            if dx > 0:
                return 'RIGHT'
            else:
                return 'LEFT'
        else:
            # Move vertically
            if dy > 0:
                return 'DOWN'
            else:
                return 'UP'
    
    def visualize(self, state, step, total_reward, title="Survival MDP"):
        """
        Visualize current state
        
        Args:
            state: (x, y, health)
            step: Current step number
            total_reward: Cumulative reward
            title: Plot title
        """
        x, y, health = state
        
        plt.figure(figsize=(12, 10))
        plt.xlim(-1, self.grid_size)
        plt.ylim(-1, self.grid_size)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.2, linewidth=0.5)
        plt.xlabel('X Position', fontsize=12)
        plt.ylabel('Y Position', fontsize=12)
        
        # Title with stats
        plt.title(f'{title}\nStep: {step} | Health: {health} | Reward: {total_reward:.1f}',
                 fontsize=14, fontweight='bold')
        
        # Draw start position
        plt.scatter(self.start[0], self.start[1], s=400, c='lightblue', 
                   marker='s', edgecolors='blue', linewidths=3, 
                   label='Start', zorder=5)
        
        # Draw goal position (big star)
        plt.scatter(self.goal[0], self.goal[1], s=600, c='gold', 
                   marker='*', edgecolors='orange', linewidths=3, 
                   label='Goal', zorder=5)
        
        # Draw agents
        food_x = [a.x for a in self.agents if a.agent_type == 'food']
        food_y = [a.y for a in self.agents if a.agent_type == 'food']
        if food_x:
            plt.scatter(food_x, food_y, s=100, c='green', marker='s', 
                       alpha=0.6, label='Food (+health)')
        
        threat_x = [a.x for a in self.agents if a.agent_type == 'threat']
        threat_y = [a.y for a in self.agents if a.agent_type == 'threat']
        if threat_x:
            plt.scatter(threat_x, threat_y, s=120, c='red', marker='^', 
                       alpha=0.7, label='Threat (-health)')
        
        info_x = [a.x for a in self.agents if a.agent_type == 'info']
        info_y = [a.y for a in self.agents if a.agent_type == 'info']
        if info_x:
            plt.scatter(info_x, info_y, s=80, c='blue', marker='o', 
                       alpha=0.6, label='Info (+health)')
        
        # Draw organism (color based on health)
        if health > 70:
            org_color = 'lime'
        elif health > 40:
            org_color = 'yellow'
        elif health > 20:
            org_color = 'orange'
        else:
            org_color = 'red'
        
        plt.scatter(x, y, s=300, c=org_color, marker='o', 
                   edgecolors='black', linewidths=3, 
                   label=f'Organism (HP:{health})', zorder=10)
        
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.show()

# ==================== RUN EPISODE ====================

def run_episode(mdp, max_steps=150, visualize_every=30, starting_health=100):
    """
    Run one episode: Try to reach the goal!
    
    Args:
        mdp: The SurvivalMDP object
        max_steps: Maximum steps allowed
        visualize_every: Show plot every N steps (0 = no viz)
        starting_health: Initial health value (default: 100)
    
    Returns:
        (success, total_reward, final_health, steps_taken)
    """
    # Initialize state
    x, y = mdp.start
    health = starting_health  # Use parameter instead of constant
    state = (x, y, health)
    
    total_reward = 0.0
    status = 'continue'
    
    print(f"\n{'='*60}")
    print(f"STARTING EPISODE")
    print(f"{'='*60}\n")
    
    # Run episode
    for step in range(max_steps):
        
        # Visualize periodically
        if visualize_every > 0 and step % visualize_every == 0:
            mdp.visualize(state, step, total_reward)
        
        # Choose action using policy
        action = mdp.simple_goal_policy(state)
        
        # Take action - get next position
        next_x, next_y, _ = mdp.transition(state, action)
        next_state = (next_x, next_y, health)  # Health not updated yet
        
        # Get reward and update health
        reward, new_health, status = mdp.get_reward(state, action, next_state)
        total_reward += reward
        
        # Update state with new health
        state = (next_x, next_y, new_health)
        health = new_health
        
        # Print step info
        print(f"Step {step:3d}: ({next_x:2d},{next_y:2d}) | "
              f"Action: {action:5s} | Health: {health:3.0f} | "
              f"Reward: {reward:6.1f} | Total: {total_reward:6.1f}")
        
        # Check terminal conditions
        if status == 'goal':
            if visualize_every > 0:
                mdp.visualize(state, step, total_reward, "SUCCESS - GOAL REACHED!")
            print(f"\n{'='*60}")
            print(f"SUCCESS! Reached goal in {step+1} steps")
            print(f"Final Health: {health}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"{'='*60}\n")
            return True, total_reward, health, step+1
        
        elif status == 'dead':
            if visualize_every > 0:
                mdp.visualize(state, step, total_reward, "FAILURE - ORGANISM DIED")
            print(f"\n{'='*60}")
            print(f"FAILED! Organism died at step {step+1}")
            print(f"Position: ({next_x}, {next_y})")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"{'='*60}\n")
            return False, total_reward, 0, step+1
    
    # Ran out of steps
    if visualize_every > 0:
        mdp.visualize(state, max_steps, total_reward, "TIMEOUT - Ran out of steps")
    print(f"\n{'='*60}")
    print(f"TIMEOUT! Ran out of steps")
    print(f"Final Position: ({state[0]}, {state[1]})")
    print(f"Final Health: {health}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"{'='*60}\n")
    return False, total_reward, health, max_steps

# ==================== MAIN ====================

if __name__ == "__main__":
    
    # Create the survival MDP
    mdp = SurvivalMDP(
        grid_size=50,
        num_food=15,      # Try changing these numbers!
        num_threats=10,   # More threats = harder
        num_info=8
    )
    
    # Run one episode
    success, reward, health, steps = run_episode(
        mdp, 
        max_steps=150,
        visualize_every=30,  # Show plot every 30 steps
        starting_health=100  # Can change this to test different starting health
    )
    
    # Summary
    if success:
        print("The organism SURVIVED and reached the goal!")
    else:
        print("The organism FAILED to reach the goal.")
    
    print("\nSimulation complete!")