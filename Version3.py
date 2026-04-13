"""
Version 3: Food Visibility and Run Tracking

Food is now hidden by default — agents must reveal it before the organism can eat.
First version to save results across runs to a JSON file.
Includes animation and learning curve graph (shown after training completes).

Grid: 40x40  |  Energy: 120  |  Max steps: 150
"""

import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import gc
import json
from datetime import datetime


# ── Long-Term Memory ───────────────────────────────────────
class LongTermMemory:
    def __init__(self, parent_memory=None, size=40):
        self.size = size
        
        if parent_memory:
            self.food_map = parent_memory.food_map.copy()
            self.agent_map = parent_memory.agent_map.copy()
            self.trap_map = parent_memory.trap_map.copy()
            self.known_food_locations = parent_memory.known_food_locations.copy()
            self.generation = parent_memory.generation + 1
            print(f"\n  Organism {self.generation} inheriting memory from successful parent")
            print(f"    Known food sources: {len(self.known_food_locations)}/5")
        else:
            self.food_map = np.zeros((size, size))
            self.agent_map = np.zeros((size, size))
            self.trap_map = np.zeros((size, size))
            self.known_food_locations = set()
            self.generation = 1
            print(f"\n  Organism 1 starting with no memory")
            print(f"    Known food sources: 0/5")
    
    def get_known_food_locations(self):
        return self.known_food_locations
    
    def discover_food(self, position):
        x, y = position
        self.food_map[x][y] = 1.0
        self.known_food_locations.add(position)
    
    def discover_agent(self, position):
        x, y = position
        self.agent_map[x][y] = 1.0
    
    def discover_trap(self, position):
        x, y = position
        self.trap_map[x][y] = 1.0
    
    def clear_all_memories(self):
        self.food_map.fill(0)
        self.agent_map.fill(0)
        self.trap_map.fill(0)
    
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


# ── Visualization ──────────────────────────────────────────
def prepare_visualization_data(env, path, generation, episode):
    print(f"  Animation data prepared for Organism {generation}!")
    return {
        'env': env,
        'path': path,
        'generation': generation,
        'episode': episode,
        'size': env.size,
        'food_locations': env.food_locations.copy(),
        'final_visible_food': env.visible_food.copy(),
        'agents': [{'position': a['position']} for a in env.agents],
        'threats': env.threats.copy(),
        'end_location': env.end_location
    }


def animate_run(data):
    size = data['size']
    path = data['path']
    generation = data['generation']
    episode = data['episode']
    
    print(f"\n  Animating Organism {generation}...")
    
    fig, ax = plt.subplots(figsize=(14, 14))
    plt.ion()
    
    positions = [step['position'] for step in path]
    
    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    
    for i in range(size + 1):
        ax.axhline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.3)
    
    for food in data['food_locations']:
        if food in data['final_visible_food']:
            circle = patches.Circle(food, 0.5, color='green', alpha=0.9, zorder=2)
            ax.add_patch(circle)
            ax.text(food[0], food[1], 'F', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white', zorder=3)
        else:
            circle = patches.Circle(food, 0.5, color='gray', fill=False,
                                   linestyle='--', linewidth=2, alpha=0.3, zorder=2)
            ax.add_patch(circle)
            ax.text(food[0], food[1], '?', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='gray', alpha=0.3, zorder=3)
    
    for agent in data['agents']:
        pos = agent['position']
        square = patches.Rectangle((pos[0]-0.5, pos[1]-0.5), 1, 1,
                                   color='blue', alpha=0.8, zorder=2)
        ax.add_patch(square)
        ax.text(pos[0], pos[1], 'A', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', zorder=3)
    
    for threat in data['threats']:
        circle = patches.Circle(threat, 0.5, color='red', alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.plot(threat[0], threat[1], 'kx', markersize=15, markeredgewidth=3, zorder=3)
    
    goal = data['end_location']
    star = patches.RegularPolygon(goal, 5, radius=0.7, color='gold', alpha=0.9, zorder=2)
    ax.add_patch(star)
    ax.text(goal[0], goal[1]+1.2, 'GOAL', ha='center', va='center',
           fontsize=12, fontweight='bold', color='gold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    start = patches.Circle((0, 0), 0.5, color='cyan', alpha=0.9, zorder=2)
    ax.add_patch(start)
    ax.text(0, -1.2, 'START', ha='center', va='center',
           fontsize=12, fontweight='bold', color='cyan',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.grid(False)
    
    path_line, = ax.plot([], [], color='orange', linewidth=3, alpha=0.4, zorder=1)
    organism_dot = patches.Circle((0, 0), 0.7, color='darkorange',
                                 edgecolor='black', linewidth=3, zorder=10)
    ax.add_patch(organism_dot)
    
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                        ha='center', fontsize=16, fontweight='bold')
    energy_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    event_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    for frame in range(len(positions)):
        current_pos = positions[frame]
        current_step = path[frame]
        energy = current_step['energy']
        
        if frame > 0:
            path_x = [positions[i][0] for i in range(frame + 1)]
            path_y = [positions[i][1] for i in range(frame + 1)]
            path_line.set_data(path_x, path_y)
        
        organism_dot.center = current_pos
        
        title = f"Organism {generation} - Episode {episode} - SUCCESS!\n"
        title += f"Step {frame + 1}/{len(path)}"
        title_text.set_text(title)
        
        energy_color = 'green' if energy > 60 else 'orange' if energy > 20 else 'red'
        energy_text.set_text(f"Energy: {energy}/120")
        energy_text.get_bbox_patch().set_facecolor(energy_color)
        
        events = []
        if current_step.get('ate_food'):
            events.append("ATE FOOD!")
        if current_step.get('hit_trap'):
            events.append("HIT TRAP!")
        if current_step.get('found_agent'):
            events.append("FOUND AGENT!")
        
        if events:
            event_text.set_text("\n".join(events))
            event_text.set_visible(True)
        else:
            event_text.set_visible(False)
        
        plt.pause(0.05)
    
    plt.pause(1.0)
    plt.ioff()
    plt.show()
    print(f"  Animation complete! {len(path)} steps shown.")
    plt.close(fig)


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
        # Food starts visible only if a prior successful organism already found it
        self.visible_food = set(self.ltm.get_known_food_locations())
        self.reset()
    
    def reset(self):
        self.agent_pos = (0, 0)
        self.energy = 120
        self.reached_end = False
        self.hit_threats_count = 0
        self.working_memory = []
        self.current_path = []
        self.ltm.clear_all_memories()
        return self.get_state()
    
    def get_directions_from_agent(self, agent_idx):
        # Visiting an agent reveals their food source and loads directions into working memory
        agent = self.agents[agent_idx]
        food_pos = agent['knows_food']
        self.visible_food.add(food_pos)
        directions = self.compute_path(self.agent_pos, food_pos)
        self.working_memory = directions
        return len(directions)
    
    def compute_path(self, start, goal):
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
        is_hungry = 1.0 if self.energy < 25 else 0.0
        
        food_here = 1.0 if (self.agent_pos in self.food_locations and
                           self.agent_pos in self.visible_food) else 0.0
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
        
        return torch.tensor([
            x / self.size, y / self.size, self.energy / 120.0, is_hungry,
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
        
        # Food can only be eaten if it has been revealed by an agent
        if self.agent_pos in self.food_locations and self.agent_pos in self.visible_food:
            self.ltm.discover_food(self.agent_pos)
            if action == 4:
                self.energy = min(120, self.energy + 25)
                reward = 10.0
                step_record['ate_food'] = True
        
        if self.agent_pos == self.end_location:
            self.reached_end = True
            reward = 100.0
        
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
    def __init__(self, state_dim=25, hidden_dim=128, action_dim=5):
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
        del parent_brain
        gc.collect()
        print(f"  Organism {ltm.generation} inherited neural network")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n  Organism {ltm.generation} starting training...")
    
    episode = 0
    
    while episode < max_episodes:
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(150):
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
            
            state = next_state.detach()
            
            if done:
                break
        
        # Periodic memory cleanup to avoid buildup over long runs
        if episode % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if episode % 100 == 0:
            status = "GOAL!" if env.reached_end else "died" if env.energy <= 0 else "timeout"
            print(f"    Episode {episode}: Reward={episode_reward:.1f}, Steps={steps}, Status={status}")
        
        if env.reached_end:
            print(f"\n  Organism {ltm.generation} REACHED GOAL!")
            print(f"    Took {episode + 1} episodes, {steps} steps, {env.hit_threats_count} traps hit")
            
            viz_data = prepare_visualization_data(env, env.current_path, ltm.generation, episode + 1)
            gc.collect()
            
            return model.state_dict(), episode + 1, {
                'episodes': episode + 1,
                'final_steps': steps,
                'traps_hit': env.hit_threats_count,
                'food_found': len(env.visible_food),
                'energy_remaining': env.energy,
                'success': True
            }, viz_data
        
        episode += 1
    
    print(f"\n  Organism {ltm.generation} FAILED")
    gc.collect()
    return model.state_dict(), max_episodes, {'episodes': max_episodes, 'success': False}, None


# ── Data Tracking ──────────────────────────────────────────
def save_run_data(organisms, episodes_list):
    history_file = "version3_history.json"

    successful = [o for o in organisms if o['metrics'].get('success', True)
                  and 'final_steps' in o['metrics']]
    if not successful:
        print("  No successful organisms to save.")
        return

    steps_list  = [o['metrics']['final_steps']     for o in successful]
    food_list   = [o['metrics']['food_found']       for o in successful]
    energy_list = [o['metrics']['energy_remaining'] for o in successful]

    run_data = {
        'timestamp':      datetime.now().isoformat(),
        'version':        'Version 3',
        'num_organisms':  len(organisms),
        'num_successes':  len(successful),
        'best_steps':     min(steps_list),
        'worst_steps':    max(steps_list),
        'avg_steps':      sum(steps_list)  / len(steps_list),
        'avg_food_found': sum(food_list)   / len(food_list),
        'avg_energy':     sum(energy_list) / len(energy_list),
        'per_generation': [
            {
                'generation':       o['generation'],
                'episodes':         o['metrics']['episodes'],
                'steps':            o['metrics']['final_steps'],
                'traps_hit':        o['metrics']['traps_hit'],
                'food_found':       o['metrics']['food_found'],
                'energy_remaining': o['metrics']['energy_remaining'],
            }
            for o in successful
        ]
    }

    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {'runs': []}

    history['runs'].append(run_data)

    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[*] Run data saved to {history_file} (Total runs: {len(history['runs'])})")


def create_learning_curve_graph(organisms, episodes_list, avg_episodes, min_episodes, max_episodes):
    print("\n" + "-"*80)
    print("GENERATING LEARNING CURVE GRAPH...")
    print("-"*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    generations = list(range(1, len(organisms) + 1))
    
    # Left plot: current run
    ax1.plot(generations, episodes_list, marker='o', linewidth=2.5, markersize=10,
            color='#2E86AB', markerfacecolor='#A23B72', markeredgewidth=2,
            markeredgecolor='#2E86AB', label='Episodes to Goal')
    ax1.axhline(y=avg_episodes, color='orange', linestyle='--', linewidth=2,
               label=f'Average: {avg_episodes:.1f}', alpha=0.7)
    
    best_gen = episodes_list.index(min_episodes) + 1
    worst_gen = episodes_list.index(max_episodes) + 1
    ax1.plot(best_gen, min_episodes, marker='*', markersize=20, color='green',
            markeredgewidth=2, markeredgecolor='darkgreen', zorder=10)
    ax1.plot(worst_gen, max_episodes, marker='X', markersize=15, color='red',
            markeredgewidth=3, zorder=10)
    ax1.annotate(f'Best: {min_episodes}', xy=(best_gen, min_episodes),
                xytext=(best_gen, min_episodes - 15), fontsize=10, fontweight='bold',
                ha='center', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.annotate(f'Worst: {max_episodes}', xy=(worst_gen, max_episodes),
                xytext=(worst_gen, max_episodes + 15), fontsize=10, fontweight='bold',
                ha='center', color='red',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Episodes to Goal', fontsize=12, fontweight='bold')
    ax1.set_title('Current Run: Evolutionary Learning Progress', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xticks(generations)
    
    # Right plot: historical average
    history_file = "simulation_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if len(history['runs']) > 0:
            num_gens = len(episodes_list)
            avg_per_gen = [0] * num_gens
            num_valid_runs = 0

            for run in history['runs']:
                # FIX: skip old entries that don't have episodes_per_generation
                run_episodes = run.get('episodes_per_generation', [])
                if not run_episodes:
                    continue
                if len(run_episodes) == num_gens:
                    for i, eps in enumerate(run_episodes):
                        avg_per_gen[i] += eps
                    num_valid_runs += 1

            if num_valid_runs > 0:
                avg_per_gen = [total / num_valid_runs for total in avg_per_gen]
                ax2.plot(generations, avg_per_gen, marker='s', linewidth=2.5, markersize=8,
                        color='#6A4C93', markerfacecolor='#F4A261', markeredgewidth=2,
                        markeredgecolor='#6A4C93', label=f'Avg across {num_valid_runs} runs')
                ax2.plot(generations, episodes_list, marker='o', linewidth=1.5, markersize=6,
                        color='#2E86AB', alpha=0.5, linestyle='--', label='Current run')
                ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Average Episodes to Goal', fontsize=12, fontweight='bold')
                ax2.set_title(f'Historical Performance ({num_valid_runs} runs)',
                             fontsize=13, fontweight='bold', pad=15)
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.legend(loc='upper right', fontsize=10)
                ax2.set_xticks(generations)
            else:
                ax2.text(0.5, 0.5, 'No compatible historical data\n(Run more simulations)',
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Historical Performance', fontsize=13, fontweight='bold', pad=15)
        else:
            ax2.text(0.5, 0.5, 'First run - no history yet',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Historical Performance', fontsize=13, fontweight='bold', pad=15)
    else:
        ax2.text(0.5, 0.5, 'First run - no history yet',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Historical Performance', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.show()
    print("[*] Learning curve graphs displayed!")
    print("-"*80)
    return fig


# ── Multi-Organism Runner ──────────────────────────────────
def train_multi_organism(num_organisms=3, max_episodes_per_organism=500):
    print("="*70)
    print("VERSION 3 WITH VISUALIZATION")
    print("="*70)
    
    viz_data_list = []
    
    ltm = LongTermMemory(parent_memory=None)
    brain, episodes, metrics, viz_data = train_organism(ltm, parent_brain=None, max_episodes=max_episodes_per_organism)
    if viz_data:
        viz_data_list.append(viz_data)
    
    organisms = [{'generation': 1, 'brain': brain, 'ltm': ltm, 'metrics': metrics}]
    
    for gen in range(2, num_organisms + 1):
        parent = organisms[-1]
        
        # Only inherit LTM from successful parents — failed runs leave nothing behind
        if parent['metrics'].get('success', True):
            ltm = LongTermMemory(parent_memory=parent['ltm'])
        else:
            ltm = LongTermMemory(parent_memory=None)
        
        brain, episodes, metrics, viz_data = train_organism(
            ltm, parent_brain=parent['brain'], max_episodes=max_episodes_per_organism
        )
        if viz_data:
            viz_data_list.append(viz_data)
        organisms.append({'generation': gen, 'brain': brain, 'ltm': ltm, 'metrics': metrics})
    
    print("\n" + "="*80)
    print("EVOLUTIONARY PROGRESS SUMMARY")
    print("="*80)
    
    episodes_list = [org['metrics']['episodes'] for org in organisms]
    max_episodes = max(episodes_list)
    min_episodes = min(episodes_list)
    avg_episodes = sum(episodes_list) / len(episodes_list)
    
    print("\n" + "-"*80)
    print(f"{'Gen':<6} {'Episodes':<12} {'Change':<12}")
    print("-"*80)
    
    for i, org in enumerate(organisms):
        gen = org['generation']
        eps = org['metrics']['episodes']
        if i == 0:
            change = "baseline"
        else:
            prev_eps = organisms[i-1]['metrics']['episodes']
            diff = eps - prev_eps
            if diff > 0:
                change = f"+{diff} worse"
            elif diff < 0:
                change = f"{diff} better"
            else:
                change = "same"
        print(f"{gen:<6} {eps:<12} {change:<12}")
    
    print("-"*80)
    print(f"\nStatistics:")
    print(f"  Best Performance:    {min_episodes} episodes (Generation {episodes_list.index(min_episodes) + 1})")
    print(f"  Worst Performance:   {max_episodes} episodes (Generation {episodes_list.index(max_episodes) + 1})")
    print(f"  Average Performance: {avg_episodes:.1f} episodes")
    print(f"  Improvement:         {episodes_list[0] - min_episodes} episodes ({((episodes_list[0] - min_episodes) / episodes_list[0] * 100):.1f}% reduction)")
    
    save_run_data(organisms, episodes_list)
    
    if viz_data_list:
        print("\n" + "="*70)
        print("REPLAY MENU")
        print("="*70)
        
        while True:
            print("\n" + "="*40)
            print("Choose what to replay:")
            print("="*40)
            print(f"  G. Show graphs again")
            for i, viz_data in enumerate(viz_data_list, 1):
                gen = viz_data['generation']
                ep = viz_data['episode']
                steps = len(viz_data['path'])
                metrics = organisms[gen-1]['metrics']
                episodes_to_goal = metrics['episodes']
                print(f"  {i}. Organism {gen} - {episodes_to_goal} episodes to goal, {steps} steps")
            print(f"  0. Exit replay menu")
            print("="*40)
            
            try:
                choice = input("\nEnter choice (0-{} or G): ".format(len(viz_data_list)))
                if choice.upper() == 'G':
                    print("\n[>] Showing graphs again...")
                    create_learning_curve_graph(organisms, episodes_list, avg_episodes, min_episodes, max_episodes)
                elif choice == '0':
                    print("\n[*] Exiting replay menu. Thanks for watching!")
                    break
                else:
                    choice = int(choice)
                    if 1 <= choice <= len(viz_data_list):
                        print(f"\n[>] Replaying Organism {viz_data_list[choice-1]['generation']}...")
                        animate_run(viz_data_list[choice-1])
                    else:
                        print(f"[!] Invalid choice. Please enter 0-{len(viz_data_list)} or G")
            except ValueError:
                print("[!] Invalid input. Please enter a number or G.")
            except KeyboardInterrupt:
                print("\n\n[*] Exiting replay menu.")
                break
    
    return organisms, viz_data_list


if __name__ == "__main__":
    try:
        print("\n[*] Starting organism evolution simulation...")
        print("Animations will be shown AFTER all training completes.\n")
        
        organisms, viz_data_list = train_multi_organism(num_organisms=10, max_episodes_per_organism=1000)
        print(f"\n[*] Training completed successfully!")
        print(f"[*] Generated {len(viz_data_list)} animations")
    except Exception as e:
        print(f"\n[!] Error occurred: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nMemory cleaned up after error.")