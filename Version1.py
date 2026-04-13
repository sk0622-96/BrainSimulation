"""
Version 1: Survival Grid MDP

50x50 grid, organism navigates from (0,0) to (49,49) using a greedy policy.
No learning, no memory. Establishes the MDP structure used in all later versions.

Agent types:
- Food:   +20 health
- Threat: -30 health
- Info:   +5 health
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────
GRID_SIZE       = 50
START_POSITION  = (0, 0)
GOAL_POSITION   = (49, 49)

FOOD_HEALTH     = 20
THREAT_HEALTH   = -30
INFO_HEALTH     = 5

GOAL_REWARD     = 100.0
DEATH_PENALTY   = -50.0
MOVE_COST       = -0.1

STARTING_HEALTH = 100
ACTIONS         = ['UP', 'DOWN', 'LEFT', 'RIGHT']


# ── Agent ──────────────────────────────────────────────────
class Agent:
    def __init__(self, x, y, agent_type, health_change):
        self.x            = x
        self.y            = y
        self.agent_type   = agent_type
        self.health_change = health_change


# ── MDP ────────────────────────────────────────────────────
class SurvivalMDP:
    """
    State: (x, y, health)
    Goal:  Reach (49, 49) with health > 0
    """

    def __init__(self, grid_size=50, num_food=15, num_threats=10, num_info=8):
        self.grid_size     = grid_size
        self.start         = START_POSITION
        self.goal          = GOAL_POSITION
        self.agents        = []
        self.initial_agents = []

        np.random.seed(42)

        # Place agents, avoiding start and goal
        for agent_type, health, count in [
            ('food',   FOOD_HEALTH,   num_food),
            ('threat', THREAT_HEALTH, num_threats),
            ('info',   INFO_HEALTH,   num_info),
        ]:
            for _ in range(count):
                while True:
                    x = np.random.randint(0, grid_size)
                    y = np.random.randint(0, grid_size)
                    if (x, y) != self.start and (x, y) != self.goal:
                        self.agents.append(Agent(x, y, agent_type, health))
                        break

        self.initial_agents = [Agent(a.x, a.y, a.agent_type, a.health_change)
                               for a in self.agents]

        print(f"\n{'='*60}")
        print(f"SURVIVAL MDP: {grid_size}x{grid_size} Grid")
        print(f"{'='*60}")
        print(f"Start: {self.start}  |  Goal: {self.goal}")
        print(f"Starting Health: {STARTING_HEALTH}")
        print(f"Food: {num_food}  Threats: {num_threats}  Info: {num_info}")
        print(f"{'='*60}\n")

    def reset(self):
        self.agents = [Agent(a.x, a.y, a.agent_type, a.health_change)
                      for a in self.initial_agents]

    def get_agent_at(self, x, y):
        for agent in self.agents:
            if agent.x == x and agent.y == y:
                return agent
        return None

    def is_valid_position(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_goal(self, x, y):
        return (x, y) == self.goal

    def transition(self, state, action):
        x, y, health = state
        moves = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
        dx, dy = moves.get(action, (0, 0))
        new_x, new_y = x + dx, y + dy
        if not self.is_valid_position(new_x, new_y):
            new_x, new_y = x, y
        return (new_x, new_y, health)

    def get_reward(self, state, action, next_state):
        x, y, health = next_state
        reward     = MOVE_COST
        new_health = health
        status     = 'continue'

        if self.is_goal(x, y):
            print(f"    GOAL REACHED! Health remaining: {new_health}")
            return reward + GOAL_REWARD, new_health, 'goal'

        agent = self.get_agent_at(x, y)
        if agent is not None:
            new_health += agent.health_change
            reward     += agent.health_change * 0.5
            print(f"    {agent.agent_type.capitalize()} encountered! "
                  f"Health: {health} -> {new_health}")
            self.agents.remove(agent)

        if new_health <= 0:
            new_health = 0
            status     = 'dead'
            print(f"    ORGANISM DIED!")
            return reward + DEATH_PENALTY, new_health, status

        return reward, new_health, status

    def simple_goal_policy(self, state):
        """Greedy policy: move toward goal along the longer axis."""
        x, y, _ = state
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        if abs(dx) > abs(dy):
            return 'RIGHT' if dx > 0 else 'LEFT'
        else:
            return 'DOWN' if dy > 0 else 'UP'

    def visualize(self, state, step, total_reward, title="Survival MDP"):
        x, y, health = state
        plt.figure(figsize=(12, 10))
        plt.xlim(-1, self.grid_size)
        plt.ylim(-1, self.grid_size)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.2, linewidth=0.5)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'{title}\nStep: {step} | Health: {health} | Reward: {total_reward:.1f}',
                  fontsize=14, fontweight='bold')

        plt.scatter(*self.start, s=400, c='lightblue', marker='s',
                    edgecolors='blue', linewidths=3, label='Start', zorder=5)
        plt.scatter(*self.goal, s=600, c='gold', marker='*',
                    edgecolors='orange', linewidths=3, label='Goal', zorder=5)

        for atype, color, marker, label in [
            ('food',   'green', 's', 'Food (+health)'),
            ('threat', 'red',   '^', 'Threat (-health)'),
            ('info',   'blue',  'o', 'Info (+health)'),
        ]:
            ax = [a.x for a in self.agents if a.agent_type == atype]
            ay = [a.y for a in self.agents if a.agent_type == atype]
            if ax:
                plt.scatter(ax, ay, s=100, c=color, marker=marker,
                            alpha=0.6, label=label)

        org_color = ('lime' if health > 70 else 'yellow' if health > 40
                     else 'orange' if health > 20 else 'red')
        plt.scatter(x, y, s=300, c=org_color, marker='o',
                    edgecolors='black', linewidths=3,
                    label=f'Organism (HP:{health})', zorder=10)

        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()


# ── Episode runner ─────────────────────────────────────────
def run_episode(mdp, max_steps=150, visualize_every=30, starting_health=100):
    x, y   = mdp.start
    state  = (x, y, starting_health)
    health = starting_health
    total_reward = 0.0

    print(f"\n{'='*60}\nSTARTING EPISODE\n{'='*60}\n")

    for step in range(max_steps):
        if visualize_every > 0 and step % visualize_every == 0:
            mdp.visualize(state, step, total_reward)

        action              = mdp.simple_goal_policy(state)
        next_x, next_y, _  = mdp.transition(state, action)
        next_state          = (next_x, next_y, health)
        reward, new_health, status = mdp.get_reward(state, action, next_state)

        total_reward += reward
        state  = (next_x, next_y, new_health)
        health = new_health

        print(f"Step {step:3d}: ({next_x:2d},{next_y:2d}) | "
              f"Action: {action:5s} | Health: {health:3.0f} | "
              f"Reward: {reward:6.1f} | Total: {total_reward:6.1f}")

        if status == 'goal':
            if visualize_every > 0:
                mdp.visualize(state, step, total_reward, "SUCCESS - GOAL REACHED!")
            print(f"\n{'='*60}\nSUCCESS! Reached goal in {step+1} steps"
                  f"\nFinal Health: {health}\nTotal Reward: {total_reward:.2f}\n{'='*60}\n")
            return True, total_reward, health, step + 1

        elif status == 'dead':
            if visualize_every > 0:
                mdp.visualize(state, step, total_reward, "FAILURE - ORGANISM DIED")
            print(f"\n{'='*60}\nFAILED! Organism died at step {step+1}"
                  f"\nTotal Reward: {total_reward:.2f}\n{'='*60}\n")
            return False, total_reward, 0, step + 1

    if visualize_every > 0:
        mdp.visualize(state, max_steps, total_reward, "TIMEOUT")
    print(f"\n{'='*60}\nTIMEOUT!\nFinal Position: ({state[0]}, {state[1]})"
          f"\nFinal Health: {health}\nTotal Reward: {total_reward:.2f}\n{'='*60}\n")
    return False, total_reward, health, max_steps


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    mdp = SurvivalMDP(grid_size=50, num_food=15, num_threats=10, num_info=8)

    success, reward, health, steps = run_episode(
        mdp,
        max_steps=150,
        visualize_every=30,
        starting_health=100
    )

    print("SUCCESS: organism reached the goal." if success
          else "FAILED: organism did not reach the goal.")
    print("Simulation complete!")