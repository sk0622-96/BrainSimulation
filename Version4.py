"""
VERSION 6: Auto-eat + Trap Repositioning + Conflict Fix
=========================================================
- Auto-eat: organism eats automatically when standing on visible food
- Traps repositioned: removed (12,12) conflict with agent, moved
  (10,10) and (30,30) off diagonal to edges
- All food, agent, trap positions verified no overlaps
- Correct memory inheritance from Version 5 retained

Memory flow:
- Each episode resets to inherited_food only (failed = forgotten)
- On success: save_successful_run() locks in what gets passed on
- Next generation inherits ONLY successful_food_locations

Grid: 40x40
"""

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
import os
import time
from datetime import datetime
import random
import re
from huggingface_hub import InferenceClient

# ==================== HF BRAIN SETUP ====================
HF_TOKEN = "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"  # paste your token here
hf_client = InferenceClient(api_key=HF_TOKEN)

# Models in priority order — will auto-switch if one fails/quota hits
HF_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
]
active_hf_model = [0]  # list so it's mutable inside nested functions

# Track brain usage stats
brain_stats = {
    'hf_calls': 0,
    'fallback_calls': 0,
    'model_usage': {m: 0 for m in HF_MODELS},
    'fallback': 0
}


# ==================== NATURAL LANGUAGE COMMUNICATION ====================
DIRECTION_TEMPLATES = [
    "You can find a food source when you make {v_num} steps {v_dir} and {h_num} steps to the {h_dir}",
    "A food source becomes accessible if you move {v_num} steps {v_dir} and then {h_num} steps to the {h_dir}",
    "You'll reach food by taking {v_num} steps {v_dir} followed by {h_num} steps to the {h_dir}",
    "Move {v_dir} {v_num} steps and shift {h_dir} {h_num} steps to locate a food source",
    "You can locate food after advancing {v_num} steps {v_dir} and turning {h_dir} for {h_num} steps",
    "Take {v_num} steps {v_dir}, then {h_num} to the {h_dir}, to find a source of food",
    "A supply of food can be found by going {v_num} steps {v_dir} and {h_num} steps {h_dir}",
    "You will discover food once you step {v_dir} {v_num} times and then move {h_dir} {h_num} times",
    "Food is reachable if you proceed {v_num} steps {v_dir} and move {h_num} steps to your {h_dir}",
    "Step {v_dir} {v_num} times and then take {h_num} {h_dir}ward steps to reach food",
    "You can access food by advancing {v_num} steps {v_dir} and veering {h_dir} for {h_num} steps",
    "A food source lies {v_num} steps {v_dir} of you and {h_num} steps to the {h_dir}",
    "Walk {v_num} steps {v_dir}, then shift {h_num} steps {h_dir} to find food",
    "By moving {v_num} paces {v_dir} and {h_num} paces {h_dir}, you can find a food source",
    "Go {v_dir} {v_num} steps, then head {h_dir} for {h_num} steps to locate food"
]


def generate_agent_response(up_steps, down_steps, left_steps, right_steps):
    """Generate natural language with correct directional words"""
    if up_steps > 0:
        v_dir = "forward"
        v_num = up_steps
    elif down_steps > 0:
        v_dir = "backward"
        v_num = down_steps
    else:
        v_dir = "forward"
        v_num = 0

    if right_steps > 0:
        h_dir = "right"
        h_num = right_steps
    elif left_steps > 0:
        h_dir = "left"
        h_num = left_steps
    else:
        h_dir = "right"
        h_num = 0

    template = random.choice(DIRECTION_TEMPLATES)
    response = template.format(v_dir=v_dir, v_num=v_num, h_dir=h_dir, h_num=h_num)
    return response


def _regex_fallback(agent_response):
    """Emergency regex fallback parser — used only when all HF models fail"""
    numbers = re.findall(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b',
                         agent_response, re.IGNORECASE)
    word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                   'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

    parsed_numbers = []
    for num in numbers:
        if num.isdigit():
            parsed_numbers.append(int(num))
        else:
            parsed_numbers.append(word_to_num.get(num.lower(), 0))

    response_lower = agent_response.lower()

    up_steps = 0
    if 'forward' in response_lower or 'ahead' in response_lower or 'straight' in response_lower:
        up_steps = parsed_numbers[0] if len(parsed_numbers) > 0 else 0
    elif 'backward' in response_lower or 'back' in response_lower:
        up_steps = -(parsed_numbers[0] if len(parsed_numbers) > 0 else 0)

    right_steps = 0
    if 'right' in response_lower or 'east' in response_lower:
        right_steps = parsed_numbers[1] if len(parsed_numbers) > 1 else 0
    elif 'left' in response_lower or 'west' in response_lower:
        right_steps = -(parsed_numbers[1] if len(parsed_numbers) > 1 else 0)

    return {'up': up_steps, 'right': right_steps}


def parse_directions_hf_brain(agent_response):
    """
    THE ORGANISM'S BRAIN — uses a real LLM to understand natural language directions.
    Tries HF models in priority order, auto-switching on quota/errors.
    Falls back to regex only as last resort.
    """
    global active_hf_model, brain_stats

    for offset in range(len(HF_MODELS)):
        idx = (active_hf_model[0] + offset) % len(HF_MODELS)
        model = HF_MODELS[idx]

        try:
            out = hf_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Convert this navigation instruction into JSON.\n"
                        "Rules:\n"
                        "- forward/ahead/straight => up positive\n"
                        "- backward/back => up negative\n"
                        "- right/east => right positive\n"
                        "- left/west => right negative\n"
                        "- include both fields even if 0\n"
                        'Return ONLY a JSON object, no explanation. Example: {"up": 3, "right": -2}\n\n'
                        f"Instruction: {agent_response}"
                    )
                }],
                max_tokens=32,
                temperature=0.1,
            )
            raw = out.choices[0].message.content.strip()

            if idx != active_hf_model[0]:
                active_hf_model[0] = idx

            m = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                if "up" in obj and "right" in obj:
                    brain_stats['hf_calls'] += 1
                    brain_stats['model_usage'][model] = brain_stats['model_usage'].get(model, 0) + 1
                    return {'up': int(obj["up"]), 'right': int(obj["right"])}

            break

        except Exception as e:
            err = str(e).lower()
            is_quota = any(x in err for x in ["402", "429", "rate limit", "quota", "payment", "credit"])
            is_unavailable = any(x in err for x in ["404", "410", "503", "unavailable", "gone"])

            if is_quota:
                time.sleep(0.5)
                continue
            elif is_unavailable:
                continue
            else:
                break

    brain_stats['fallback_calls'] += 1
    return _regex_fallback(agent_response)


# ==================== LONG-TERM MEMORY ====================
class LongTermMemory:
    def __init__(self, parent_memory=None, size=40):
        self.size = size

        if parent_memory:
            self.inherited_food = parent_memory.successful_food_locations.copy()
            self.inherited_traps = parent_memory.successful_trap_locations.copy()
            self.generation = parent_memory.generation + 1
        else:
            self.inherited_food = set()
            self.inherited_traps = set()
            self.generation = 1

        self.food_map = np.zeros((size, size))
        self.agent_map = np.zeros((size, size))
        self.trap_map = np.zeros((size, size))

        # Mark inherited traps in trap_map so organism avoids them from the start
        for (x, y) in self.inherited_traps:
            self.trap_map[x][y] = 1.0

        self.known_food_locations = self.inherited_food.copy()
        self.known_trap_locations = self.inherited_traps.copy()

        self.successful_food_locations = set()
        self.successful_trap_locations = set()

    def get_known_food_locations(self):
        return self.inherited_food  # GridWorld uses this to set inherited_food at init

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
        self.known_trap_locations.add(position)

    def save_successful_run(self):
        """Locks in food and traps found on this run to pass to the next organism"""
        self.successful_food_locations = self.known_food_locations.copy()
        self.successful_trap_locations = self.known_trap_locations.copy()

    def reset_episode(self):
        """Called on every reset — wipes episode discoveries, keeps only inherited"""
        self.food_map.fill(0)
        self.agent_map.fill(0)
        self.trap_map.fill(0)
        self.known_food_locations = self.inherited_food.copy()
        self.known_trap_locations = self.inherited_traps.copy()
        # Re-mark inherited traps in trap_map
        for (x, y) in self.inherited_traps:
            self.trap_map[x][y] = 1.0

    def clear_all_memories(self):
        self.reset_episode()

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


# ==================== VISUALIZATION ====================
def prepare_visualization_data(env, path, generation, episode):
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
    """Animate a successful run from stored data"""
    size = data['size']
    path = data['path']
    organism_num = data.get('organism_num', data['generation'])

    print(f"\n  Animating Organism {organism_num}...")

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

        title = f"Organism {organism_num} - SUCCESS!\n"
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
    plt.close(fig)


# ==================== ENVIRONMENT ====================
class GridWorld:
    def __init__(self, ltm, size=40):
        self.size = size
        self.ltm = ltm

        self.food_locations = [
            (8, 8), (32, 12), (20, 28), (12, 36), (28, 20),
            (15, 15),
            (25, 25)
        ]

        self.agents = [
            {'position': (4, 4),   'knows_food': (8, 8)},
            {'position': (36, 8),  'knows_food': (32, 12)},
            {'position': (16, 24), 'knows_food': (20, 28)},
            {'position': (8, 32),  'knows_food': (12, 36)},
            {'position': (24, 16), 'knows_food': (28, 20)},
            {'position': (12, 12), 'knows_food': (15, 15)},
            {'position': (22, 22), 'knows_food': (25, 25)}
        ]

        self.threats = [
            (24, 8),   # kept
            (16, 20),  # kept
            (32, 32),  # kept
            (20, 36),  # kept
            (2, 36),   # was (12,12) — agent/trap conflict fixed, moved to edge
            (38, 4),   # was (10,10) — moved off diagonal to edge
            (6, 34)    # was (30,30) — moved off diagonal to edge
        ]

        self.end_location = (39, 39)
        self.current_path = []

        self.inherited_food = set(self.ltm.get_known_food_locations())
        self.visible_food = self.inherited_food.copy()
        self.communicated_agents = set()

        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.energy = 100
        self.reached_end = False
        self.hit_threats_count = 0
        self.current_path = []
        self.eaten_food = set()  # food can only be eaten once per episode
        self.visible_food = self.inherited_food.copy()
        self.communicated_agents = set()
        self.ltm.reset_episode()
        return self.get_state()

    def communicate_with_agent(self, agent_idx):
        """
        LINGUISTIC COMMUNICATION SYSTEM
        1. Agent generates varied natural language directions
        2. Organism's HF brain parses the language
        3. Execute directions
        4. Reveal food location
        """
        agent = self.agents[agent_idx]
        food_pos = agent['knows_food']

        curr_x, curr_y = self.agent_pos
        food_x, food_y = food_pos

        up_steps = food_y - curr_y
        right_steps = food_x - curr_x

        up = up_steps if up_steps > 0 else 0
        down = abs(up_steps) if up_steps < 0 else 0
        right = right_steps if right_steps > 0 else 0
        left = abs(right_steps) if right_steps < 0 else 0

        # Agent speaks in natural language
        agent_response = generate_agent_response(up, down, left, right)
        parsed = parse_directions_hf_brain(agent_response)

        self.visible_food.add(food_pos)
        self.communicated_agents.add(agent_idx)
        self.ltm.discover_food(food_pos)

        return parsed

    def execute_linguistic_directions(self, parsed_directions):
        """Execute parsed directions — organism follows instructions from brain"""
        moves_taken = []

        vertical_moves = parsed_directions['up']
        if vertical_moves > 0:
            for _ in range(vertical_moves):
                if self.agent_pos[1] < self.size - 1 and self.energy > 0:
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
                    self.energy -= 1
                    moves_taken.append('up')
                    if self.energy <= 0:
                        break
        elif vertical_moves < 0:
            for _ in range(abs(vertical_moves)):
                if self.agent_pos[1] > 0 and self.energy > 0:
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
                    self.energy -= 1
                    moves_taken.append('down')
                    if self.energy <= 0:
                        break

        horizontal_moves = parsed_directions['right']
        if horizontal_moves > 0:
            for _ in range(horizontal_moves):
                if self.agent_pos[0] < self.size - 1 and self.energy > 0:
                    self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
                    self.energy -= 1
                    moves_taken.append('right')
                    if self.energy <= 0:
                        break
        elif horizontal_moves < 0:
            for _ in range(abs(horizontal_moves)):
                if self.agent_pos[0] > 0 and self.energy > 0:
                    self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
                    self.energy -= 1
                    moves_taken.append('left')
                    if self.energy <= 0:
                        break

        return moves_taken

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

        end_dx = (self.end_location[0] - x) / self.size
        end_dy = (self.end_location[1] - y) / self.size

        return torch.tensor([
            x / self.size, y / self.size, self.energy / 100.0, is_hungry,
            food_here, agent_here, threat_here, remembered_trap_here,
            mem_food_dx, mem_food_dy, mem_food_str,
            mem_agent_dx, mem_agent_dy, mem_agent_str,
            mem_trap_dx, mem_trap_dy, mem_trap_str,
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
                if idx in self.communicated_agents:
                    continue
                if agent['knows_food'] in self.inherited_food:
                    continue

                self.ltm.discover_agent(self.agent_pos)

                # LINGUISTIC COMMUNICATION — brain parses directions
                parsed_directions = self.communicate_with_agent(idx)
                self.execute_linguistic_directions(parsed_directions)

                reward += 5.0
                step_record['found_agent'] = True
                break

        if self.agent_pos in self.threats:
            self.ltm.discover_trap(self.agent_pos)
            self.energy -= 10
            reward = -5.0
            self.hit_threats_count += 1
            step_record['hit_trap'] = True

        if (self.agent_pos in self.food_locations
                and self.agent_pos in self.visible_food
                and self.agent_pos not in self.eaten_food):
            self.eaten_food.add(self.agent_pos)
            self.ltm.discover_food(self.agent_pos)
            self.energy = min(150, self.energy + 40)
            reward = 5.0
            step_record['ate_food'] = True

        if self.agent_pos == self.end_location:
            self.reached_end = True
            reward = 300.0

        self.current_path.append(step_record)

        done = self.energy <= 0 or self.reached_end
        if self.energy <= 0:
            reward = -100

        return self.get_state(), reward, done


# ==================== NEURAL NETWORK ====================
class NeuralNetwork(nn.Module):
    def __init__(self, state_dim=19, hidden_dim=128, action_dim=4):
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
            return np.random.randint(0, 4)
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


# ==================== TRAINING ====================
def train_organism(ltm, parent_brain=None, organism_offset=0):
    """
    Runs one organism across unlimited attempts until it reaches the goal.
    Brain trains on every attempt — failed or not.
    organism_offset is the cumulative count before this organism started.
    Returns the organism number it succeeded on (cumulative).
    """
    env = GridWorld(ltm)
    model = NeuralNetwork()

    if parent_brain:
        model.load_state_dict(copy.deepcopy(parent_brain))
        del parent_brain
        gc.collect()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attempt = 0

    while True:
        state = env.reset()
        steps = 0

        if attempt % 50 == 0 and attempt > 0:
            print(f"    ... attempt {organism_offset + attempt}, still searching...")

        for step in range(200):
            epsilon = max(0.1, 0.6 - attempt / 300)
            action = model.get_action(state, epsilon=epsilon)

            next_state, reward, done = env.step(action)
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

        attempt += 1

        if attempt % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if env.reached_end:
            organism_num = organism_offset + attempt
            env.ltm.save_successful_run()
            viz_data = prepare_visualization_data(env, env.current_path, ltm.generation, attempt)
            viz_data['organism_num'] = organism_num
            gc.collect()

            return model.state_dict(), attempt, {
                'attempts': attempt,
                'organism_num': organism_num,
                'final_steps': steps,
                'traps_hit': env.hit_threats_count,
                'food_found': len(env.visible_food),
                'inherited': len(ltm.inherited_food),
                'energy_remaining': env.energy,
                'success': True,
            }, viz_data


# ==================== DATA TRACKING & VISUALIZATION ====================
def save_run_data(organisms, episodes_list):
    history_file = "/Users/tommysupey/Desktop/Brain Simulation/version4_history.json"

    steps_list  = [o['metrics']['final_steps']     for o in organisms]
    energy_list = [o['metrics']['energy_remaining'] for o in organisms]
    food_list   = [o['metrics']['food_found']       for o in organisms]

    run_data = {
        'timestamp':      datetime.now().isoformat(),
        'organism_numbers': episodes_list,
        'best':           min(episodes_list),
        'worst':          max(episodes_list),
        'average':        sum(episodes_list) / len(episodes_list),
        'best_steps':     min(steps_list),
        'worst_steps':    max(steps_list),
        'avg_steps':      sum(steps_list)  / len(steps_list),
        'avg_energy':     sum(energy_list) / len(energy_list),
        'avg_food':       sum(food_list)   / len(food_list),
        'per_success': [
            {
                'success_num':      o['success_num'],
                'organism_num':     o['organism_num'],
                'attempts':         o['attempts'],
                'steps':            o['metrics']['final_steps'],
                'traps_hit':        o['metrics']['traps_hit'],
                'food_found':       o['metrics']['food_found'],
                'inherited':        o['metrics']['inherited'],
                'energy_remaining': o['metrics']['energy_remaining'],
            }
            for o in organisms
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

    print(f"\n  run data saved to {history_file} (total runs: {len(history['runs'])})")


def create_learning_curve_graph(organisms):
    success_nums = [o['success_num'] for o in organisms]
    organism_nums = [o['organism_num'] for o in organisms]
    steps_list = [o['metrics']['final_steps'] for o in organisms]
    inherited_list = [o['metrics']['inherited'] for o in organisms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: organism number of each success — gaps shrinking = knowledge helping
    ax1.bar(success_nums, organism_nums, color='#2E86AB', alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.plot(success_nums, organism_nums, marker='o', color='#A23B72', linewidth=2,
             markersize=8, markeredgewidth=2, markeredgecolor='white', zorder=5)

    for i, (s, o) in enumerate(zip(success_nums, organism_nums)):
        ax1.annotate(str(o), xy=(s, o), xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=9, fontweight='bold', color='#2E86AB')

    ax1.set_xlabel('Success #', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Organism Number', fontsize=12, fontweight='bold')
    ax1.set_title('Which Organism Each Success Occurred At', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_xticks(success_nums)

    # Right: steps per success, colored by how much food was inherited
    colors = plt.cm.YlGn([i / 7 for i in inherited_list])
    bars = ax2.bar(success_nums, steps_list, color=colors, edgecolor='white', linewidth=1.5)
    ax2.plot(success_nums, steps_list, marker='o', color='#E76F51', linewidth=2,
             markersize=8, markeredgewidth=2, markeredgecolor='white', zorder=5)

    avg_steps = sum(steps_list) / len(steps_list)
    ax2.axhline(y=avg_steps, color='orange', linestyle='--', linewidth=2,
                label=f'Average: {avg_steps:.0f} steps', alpha=0.8)

    for i, (s, st, inh) in enumerate(zip(success_nums, steps_list, inherited_list)):
        ax2.annotate(f'{st}\n({inh} inh)', xy=(s, st), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=8,
                     fontweight='bold', color='#333333')

    ax2.set_xlabel('Success #', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Steps to Goal', fontsize=12, fontweight='bold')
    ax2.set_title('Steps per Success (shading = food inherited)', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xticks(success_nums)

    plt.suptitle('Evolutionary Learning Progress', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return fig


# ==================== MAIN TRAINING LOOP ====================
def train_multi_organism(target_successes=10):
    print("="*70)
    print("ORGANISM EVOLUTION SIMULATION")
    print("="*70)
    print(f"\n  Running until {target_successes} successful organisms.")
    print(f"  Brain: {HF_MODELS[active_hf_model[0]]}")
    print("="*70 + "\n")

    viz_data_list = []
    organisms = []
    total_organisms = 0
    successes = 0
    last_brain = None
    last_successful_ltm = None
    ltm = LongTermMemory(parent_memory=None)

    while successes < target_successes:
        brain, attempts, metrics, viz_data = train_organism(
            ltm, parent_brain=last_brain, organism_offset=total_organisms
        )
        last_brain = brain
        total_organisms += attempts
        organism_num = total_organisms

        successes += 1
        last_successful_ltm = ltm
        if viz_data:
            viz_data_list.append(viz_data)

        print(f"  SUCCESS #{successes}  —  Organism {organism_num}  —  "
              f"{metrics['final_steps']} steps, "
              f"{metrics['food_found']} food found, "
              f"{metrics['traps_hit']} traps hit, "
              f"{metrics['inherited']} inherited")

        organisms.append({
            'success_num': successes,
            'organism_num': organism_num,
            'attempts': attempts,
            'metrics': metrics
        })

        ltm = LongTermMemory(parent_memory=last_successful_ltm)

    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)

    print(f"\n  Total organisms:  {total_organisms}")
    print(f"  Total successes:  {successes}")
    print(f"  Success rate:     {successes/total_organisms:.1%}")

    print(f"\n  {'#':<6} {'Organism':<12} {'Attempts':<12} {'Steps':<10} {'Traps':<8} {'Food':<8} {'Inherited'}")
    print(f"  {'-'*62}")
    for org in organisms:
        m = org['metrics']
        print(f"  {org['success_num']:<6} {org['organism_num']:<12} "
              f"{org['attempts']:<12} {m['final_steps']:<10} "
              f"{m['traps_hit']:<8} {m['food_found']:<8} {m['inherited']}")

    steps_list = [o['metrics']['final_steps'] for o in organisms]
    best = organisms[steps_list.index(min(steps_list))]
    worst = organisms[steps_list.index(max(steps_list))]
    print(f"\n  Fewest steps: {min(steps_list)} (Organism {best['organism_num']})")
    print(f"  Most steps:   {max(steps_list)} (Organism {worst['organism_num']})")
    print(f"  Average:      {sum(steps_list)/len(steps_list):.1f} steps")

    total = brain_stats['hf_calls'] + brain_stats['fallback_calls']
    if total > 0:
        print(f"\n  Brain usage ({total} total parses):")
        print(f"    LLM:      {brain_stats['hf_calls']} ({brain_stats['hf_calls']/total:.1%})")
        print(f"    Fallback: {brain_stats['fallback_calls']} ({brain_stats['fallback_calls']/total:.1%})")
        for model, count in brain_stats['model_usage'].items():
            if count > 0:
                print(f"    {model}: {count}")

    org_nums = [o['organism_num'] for o in organisms]
    save_run_data(organisms, org_nums)

    if viz_data_list:
        while True:
            print("\n" + "="*40)
            print("Replay menu:")
            print("  G — graphs")
            for i, vd in enumerate(viz_data_list, 1):
                org = organisms[i-1]
                print(f"  {i} — Organism {org['organism_num']} "
                      f"(Success #{org['success_num']}, {org['metrics']['final_steps']} steps)")
            print("  0 — exit")
            print("="*40)
            try:
                choice = input(f"\nChoice (0-{len(viz_data_list)} or G): ").strip()
                if choice.upper() == 'G':
                    create_learning_curve_graph(organisms)
                elif choice == '0':
                    break
                else:
                    idx = int(choice)
                    if 1 <= idx <= len(viz_data_list):
                        animate_run(viz_data_list[idx-1])
            except (ValueError, KeyboardInterrupt):
                break

    return organisms, viz_data_list


if __name__ == "__main__":
    try:
        print("\n  Starting simulation...")
        print("  Animations shown after all organisms complete.\n")
        train_multi_organism(target_successes=10)
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()