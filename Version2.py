"""
Version 2: Memory Only on Success

40x40 grid, Q-learning neural network, fixed world layout.
LTM passes food/trap/agent maps to the next organism on success only.
Working memory holds a decaying direction queue from agent communication.

Grid: 40x40  |  Energy: 300  |  Max steps: 500
"""

import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# ── Long-Term Memory ───────────────────────────────────────
class LongTermMemory:
    def __init__(self, parent_memory=None, size=40):
        self.size = size
        
        if parent_memory:
            self.food_map = parent_memory.food_map.copy()
            self.agent_map = parent_memory.agent_map.copy()
            self.trap_map = parent_memory.trap_map.copy()
            self.generation = parent_memory.generation + 1
            print(f"\n  Organism {self.generation} inheriting memory from successful parent")
        else:
            self.food_map = np.zeros((size, size))
            self.agent_map = np.zeros((size, size))
            self.trap_map = np.zeros((size, size))
            self.generation = 1
            print(f"\n  Organism 1 starting with no memory")
    
    def discover_food(self, position):
        x, y = position
        self.food_map[x][y] = 1.0
    
    def discover_agent(self, position):
        x, y = position
        self.agent_map[x][y] = 1.0
    
    def discover_trap(self, position):
        x, y = position
        self.trap_map[x][y] = 1.0
    
    def clear_all_memories(self):
        self.food_map = np.zeros((self.size, self.size))
        self.agent_map = np.zeros((self.size, self.size))
        self.trap_map = np.zeros((self.size, self.size))
    
    def get_nearest_remembered(self, position, memory_type):
        if memory_type == 'food':
            memory_map = self.food_map
        elif memory_type == 'agent':
            memory_map = self.agent_map
        elif memory_type == 'trap':
            memory_map = self.trap_map
        else:
            return 0.0, 0.0, 0.0
        
        x, y = position
        best_dist = float('inf')
        best_pos = None
        best_strength = 0.0
        
        for mx in range(self.size):
            for my in range(self.size):
                if memory_map[mx][my] > 0.3:
                    dist = abs(mx - x) + abs(my - y)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (mx, my)
                        best_strength = memory_map[mx][my]
        
        if best_pos:
            mx, my = best_pos
            return (mx - x) / self.size, (my - y) / self.size, best_strength
        else:
            return 0.0, 0.0, 0.0


# ── Environment ────────────────────────────────────────────
class GridWorld:
    def __init__(self, ltm, size=40):
        self.size = size
        self.ltm = ltm
        
        self.food_locations = [(8, 8), (32, 12), (20, 28), (12, 36), (28, 20)]
        
        self.agents = [
            {'position': (4, 4), 'knows_food': (8, 8)},
            {'position': (36, 8), 'knows_food': (32, 12)},
            {'position': (16, 24), 'knows_food': (20, 28)},
            {'position': (8, 32), 'knows_food': (12, 36)},
            {'position': (24, 16), 'knows_food': (28, 20)}
        ]
        
        self.threats = [(12, 12), (24, 8), (16, 20), (32, 32), (20, 36)]
        
        self.end_location = (39, 39)
        self.working_memory = []
        self.wm_decay_per_step = 1
        self.current_path = []
        self.reset()
    
    def reset(self):
        self.agent_pos = (0, 0)
        self.energy = 300
        self.reached_end = False
        self.hit_threats_count = 0
        self.working_memory = []
        self.current_path = []
        self.ltm.clear_all_memories()
        return self.get_state()
    
    def get_directions_from_agent(self, agent_idx):
        # Agent computes a direct coordinate path and loads it into working memory
        agent = self.agents[agent_idx]
        food_pos = agent['knows_food']
        directions = self.compute_path(self.agent_pos, food_pos)
        self.working_memory = directions
        return len(directions)
    
    def compute_path(self, start, goal):
        # Returns a step-by-step direction list from start to goal, ending with "eat"
        x1, y1 = start
        x2, y2 = goal
        directions = []
        
        while y1 < y2:
            directions.append("up")
            y1 += 1
        while y1 > y2:
            directions.append("down")
            y1 -= 1
        while x1 < x2:
            directions.append("right")
            x1 += 1
        while x1 > x2:
            directions.append("left")
            x1 -= 1
        
        directions.append("eat")
        return directions
    
    def get_current_direction(self):
        return self.working_memory[0] if len(self.working_memory) > 0 else None
    
    def get_state(self):
        x, y = self.agent_pos
        is_hungry = 1.0 if self.energy < 50 else 0.0
        
        food_here = 1.0 if self.agent_pos in self.food_locations else 0.0
        agent_here = 1.0 if any(self.agent_pos == agent['position'] for agent in self.agents) else 0.0
        threat_here = 1.0 if self.agent_pos in self.threats else 0.0
        
        mem_food_dx, mem_food_dy, mem_food_str = self.ltm.get_nearest_remembered(self.agent_pos, 'food')
        mem_agent_dx, mem_agent_dy, mem_agent_str = self.ltm.get_nearest_remembered(self.agent_pos, 'agent')
        mem_trap_dx, mem_trap_dy, mem_trap_str = self.ltm.get_nearest_remembered(self.agent_pos, 'trap')
        
        remembered_trap_here = 1.0 if self.ltm.trap_map[x][y] > 0.3 else 0.0
        
        current_dir = self.get_current_direction()
        dir_up    = 1.0 if current_dir == "up"    else 0.0
        dir_down  = 1.0 if current_dir == "down"  else 0.0
        dir_left  = 1.0 if current_dir == "left"  else 0.0
        dir_right = 1.0 if current_dir == "right" else 0.0
        dir_eat   = 1.0 if current_dir == "eat"   else 0.0
        has_directions = 1.0 if len(self.working_memory) > 0 else 0.0
        
        end_dx = (self.end_location[0] - x) / self.size
        end_dy = (self.end_location[1] - y) / self.size
        
        # 25-element state vector: position, energy, surroundings, LTM directions, WM flags, goal direction
        return torch.tensor([
            x / self.size, y / self.size, self.energy / 300.0, is_hungry,
            food_here, agent_here, threat_here, remembered_trap_here,
            mem_food_dx, mem_food_dy, mem_food_str,
            mem_agent_dx, mem_agent_dy, mem_agent_str,
            mem_trap_dx, mem_trap_dy, mem_trap_str,
            has_directions, dir_up, dir_down, dir_left, dir_right, dir_eat,
            end_dx, end_dy
        ], dtype=torch.float32)
    
    def step(self, action):
        x, y = self.agent_pos
        
        step_record = {
            'position': (x, y),
            'action': action,
            'energy': self.energy,
            'hit_trap': False,
            'found_agent': False,
            'ate_food': False
        }
        
        # Actions: 0=down, 1=up, 2=left, 3=right, 4=eat
        if action == 0 and y < self.size - 1:
            y += 1
        elif action == 1 and y > 0:
            y -= 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.size - 1:
            x += 1
        
        self.agent_pos = (x, y)
        self.energy -= 1
        reward = -0.1
        
        for idx, agent in enumerate(self.agents):
            if self.agent_pos == agent['position']:
                self.ltm.discover_agent(self.agent_pos)
                self.get_directions_from_agent(idx)
                reward += 1.0
                step_record['found_agent'] = True
        
        if self.agent_pos in self.threats:
            self.ltm.discover_trap(self.agent_pos)
            self.energy -= 10
            reward = -5.0
            self.hit_threats_count += 1
            step_record['hit_trap'] = True
        
        if self.agent_pos in self.food_locations:
            self.ltm.discover_food(self.agent_pos)
            if action == 4:
                self.energy = min(300, self.energy + 50)
                reward = 10.0
                step_record['ate_food'] = True
        
        if self.agent_pos == self.end_location:
            self.reached_end = True
            reward = 100.0
        
        # Pop one direction per step — working memory naturally decays as the organism moves
        if len(self.working_memory) > 0:
            for _ in range(self.wm_decay_per_step):
                if len(self.working_memory) > 0:
                    self.working_memory.pop(0)
        
        self.current_path.append(step_record)
        
        done = self.energy <= 0 or self.reached_end
        if self.energy <= 0:
            reward = -100
        
        return self.get_state(), reward, done


# ── Neural Network ─────────────────────────────────────────
class NeuralNetwork(nn.Module):
    """Q-network: 25 inputs, 256 hidden, 5 actions."""
    def __init__(self, state_dim=25, hidden_dim=256, action_dim=5):
        super(NeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
    
    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(0, 5)
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


# ── Training ───────────────────────────────────────────────
def train_organism(ltm, parent_brain=None, max_episodes=1000):
    env = GridWorld(ltm)
    model = NeuralNetwork()
    
    if parent_brain:
        model.load_state_dict(copy.deepcopy(parent_brain))
        print(f"  Organism {ltm.generation} inherited neural network")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    print(f"\n  Organism {ltm.generation} starting training...")
    
    episode = 0
    
    while episode < max_episodes:
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(500):
            # Epsilon decays from 0.6 to 0.1 over the first 300 episodes
            epsilon = max(0.1, 0.6 - episode / 300)
            action = model.get_action(state, epsilon=epsilon)
            
            next_state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1
            
            optimizer.zero_grad()
            q_values = model(state)
            target = q_values.clone()
            
            with torch.no_grad():
                next_q = model(next_state)
            target[action] = reward + 0.95 * torch.max(next_q)
            
            loss = nn.MSELoss()(q_values, target)
            loss.backward()
            optimizer.step()
            
            state = next_state
            
            if done:
                break
        
        if episode % 50 == 0:
            status = "GOAL!" if env.reached_end else "died" if env.energy <= 0 else "timeout"
            print(f"    Episode {episode}: Reward={episode_reward:.1f}, Steps={steps}, Status={status}")
        
        if env.reached_end:
            print(f"\n  Organism {ltm.generation} REACHED GOAL!")
            print(f"    Took {episode + 1} episodes, {steps} steps, {env.hit_threats_count} traps hit")

            return model.state_dict(), episode + 1, {
                'episodes': episode + 1,
                'final_steps': steps,
                'traps_hit': env.hit_threats_count,
                'energy_remaining': env.energy,
                'success': True
            }
        
        episode += 1
    
    print(f"\n  Organism {ltm.generation} FAILED")
    return model.state_dict(), max_episodes, {
        'episodes': max_episodes,
        'success': False
    }


import json
import os
from datetime import datetime


# ── Data Tracking ──────────────────────────────────────────
def save_run_data(organisms):
    history_file = "version2_history.json"

    successful = [o for o in organisms if o['metrics'].get('success', True)]
    if not successful:
        print("  No successful organisms to save.")
        return

    steps_list   = [o['metrics']['final_steps']     for o in successful]
    traps_list   = [o['metrics']['traps_hit']        for o in successful]
    energy_list  = [o['metrics']['energy_remaining'] for o in successful]
    episode_list = [o['metrics']['episodes']         for o in successful]

    run_data = {
        'timestamp':   datetime.now().isoformat(),
        'version':     'Version 2',
        'num_organisms': len(organisms),
        'num_successes': len(successful),
        'best_steps':  min(steps_list),
        'worst_steps': max(steps_list),
        'avg_steps':   sum(steps_list) / len(steps_list),
        'avg_energy':  sum(energy_list) / len(energy_list),
        'per_generation': [
            {
                'generation':       o['generation'],
                'episodes':         o['metrics']['episodes'],
                'steps':            o['metrics']['final_steps'],
                'traps_hit':        o['metrics']['traps_hit'],
                'energy_remaining': o['metrics']['energy_remaining'],
            }
            for o in successful
        ]
    }

    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = {'runs': []}

    history['runs'].append(run_data)

    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Run data saved to {history_file} (total runs: {len(history['runs'])})")


# ── Multi-Organism Runner ──────────────────────────────────
def train_multi_organism(num_organisms=3, max_episodes_per_organism=500):
    print("="*70)
    print("VERSION 2: Memory Only on Success")
    print("="*70)

    ltm = LongTermMemory(parent_memory=None)
    brain, episodes, metrics = train_organism(
        ltm, parent_brain=None, max_episodes=max_episodes_per_organism
    )

    organisms = [{'generation': 1, 'brain': brain, 'ltm': ltm, 'metrics': metrics}]

    for gen in range(2, num_organisms + 1):
        parent = organisms[-1]

        # Only pass LTM forward if the parent succeeded — failed runs leave nothing behind
        if parent['metrics'].get('success', True):
            ltm = LongTermMemory(parent_memory=parent['ltm'])
        else:
            ltm = LongTermMemory(parent_memory=None)

        brain, episodes, metrics = train_organism(
            ltm, parent_brain=parent['brain'], max_episodes=max_episodes_per_organism
        )
        organisms.append({'generation': gen, 'brain': brain, 'ltm': ltm, 'metrics': metrics})

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for org in organisms:
        m = org['metrics']
        if m.get('success', True):
            print(f"\n  Organism {org['generation']}: {m['episodes']} episodes, "
                  f"{m['final_steps']} steps, {m['traps_hit']} traps, "
                  f"{m['energy_remaining']} energy remaining")
        else:
            print(f"\n  Organism {org['generation']}: FAILED")

    save_run_data(organisms)
    return organisms


if __name__ == "__main__":
    organisms = train_multi_organism(num_organisms=3, max_episodes_per_organism=500)