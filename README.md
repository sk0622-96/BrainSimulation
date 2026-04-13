# Working Memory Simulations
### Thesis Project — Cognitive Memory Systems in Reinforcement Learning

This repository contains seven simulation versions exploring how working memory and long-term memory mechanisms affect organism survival in a grid-based reinforcement learning environment. Each version builds on the last, adding progressively more biologically realistic memory systems. The theoretical grounding draws from Baddeley's phonological loop, the serial position effects described by Burgess and Hitch (1999), and Mongillo's (2024) synaptic theory of working memory.

---

## Simulations

### Version 1 — `version1.py`
The original MDP. A 50x50 grid where an organism navigates from (0,0) to (49,49) using a greedy straight-line policy. No learning, no memory. Establishes the MDP structure used in all later versions.

- **Grid:** 50x50
- **Policy:** Greedy (Manhattan distance)
- **Memory:** None
- **Note:** Running this file will open matplotlib visualization windows.

---

### Version 2 — `version2.py`
Q-learning with working memory. First version to use a neural network and fixed world layout. LTM passes food, trap, and agent maps to the next organism on success only. Working memory holds a decaying direction queue received from agents.

- **Grid:** 40x40
- **Energy:** 300  |  **Max steps:** 500
- **Memory:** LTM inherited on success, WM decays each step
- **Saves to:** `version2_history.json`

---

### Version 3 — `version3.py`
Introduces food visibility — food is hidden by default and only becomes accessible after an agent reveals it. First version to save results to a JSON file across multiple runs. Includes animation of successful paths and a learning curve graph shown after training.

- **Grid:** 40x40
- **Energy:** 120  |  **Max steps:** 150
- **Memory:** LTM inherited on success, WM decays each step
- **Saves to:** `version3_history.json`
- **Note:** Running this file will open matplotlib windows for animation and graphs.

---

### Version 4 — `version4.py`
Replaces hardcoded agent directions with real natural language communication via HuggingFace. Agents generate varied direction sentences; the organism's brain (Qwen2.5-7B-Instruct) parses them via the API. Organisms are counted cumulatively across the run. Includes a regex fallback parser for when the API is unavailable.

- **Grid:** 40x40
- **Energy:** 100  |  **Max steps:** 200  |  **Food sources:** 7  |  **Agents:** 7
- **Requires:** HuggingFace API token — set `HF_TOKEN` at the top of the file
- **Saves to:** `version4_history.json`

---

### Perfect LTM — `perfect_ltm.py`
Adds full path sequence inheritance. On success, the entire action sequence is stored in `GLOBAL_BEST_PATH`. Every subsequent organism replays that path with a 5% deviation chance per step. If a shorter path is found it replaces the global best. The neural network trains continuously throughout replay. Represents the theoretical ceiling for sequential memory.

- **Grid:** 40x40
- **Energy:** 100  |  **Max steps:** 200  |  **Food sources:** 7  |  **Agents:** 7
- **Requires:** HuggingFace API token(s) — add to `HF_TOKENS` at the top of the file
- **Saves to:** `simulation_history.json`

---

### Realistic LTM — `realistic_ltm.py`
Models the serial position effect from Burgess and Hitch (1999). Only the first `PRIMACY_STEPS` (25) and last `RECENCY_STEPS` (25) actions of the best path are inherited. The middle is forgotten. Each organism replays the inherited start, Q-learns through the forgotten middle, then switches to the inherited end sequence when within `RECENCY_TRIGGER_DIST` (15) steps of the goal.

- **Grid:** 40x40
- **Energy:** 100  |  **Max steps:** 200  |  **Food sources:** 7  |  **Agents:** 7
- **Requires:** HuggingFace API token(s) — add to `HF_TOKENS` at the top of the file
- **Saves to:** `simulation_history.json`

---

### Imperfect STM — `imperfect_stm.py`
Adds a phonological loop capacity limit on top of Realistic LTM. Direction templates are expanded to 20 sentences across four word-count brackets. If a sentence exceeds `WM_CAPACITY` (20) words, random words are dropped before the brain parses it, potentially corrupting the step count or direction word. The simulation tracks total truncations and words dropped in `brain_stats`.

- **Grid:** 40x40
- **Energy:** 100  |  **Max steps:** 200  |  **Food sources:** 7  |  **Agents:** 7
- **Requires:** HuggingFace API token(s) — add to `HF_TOKENS` at the top of the file
- **Saves to:** `simulation_history.json`

---

## Requirements

```
Python 3.8+
numpy
matplotlib
torch
huggingface_hub
```

---

## Running the Simulations

Each version is a self-contained Python file. Run any version directly:

```bash
python version1.py
```

Versions 4, Perfect LTM, Realistic LTM, and Imperfect STM require a HuggingFace API token to run the Qwen language model. Add your token(s) to the `HF_TOKENS` list at the top of each file before running.

JSON history files are not included in this repository. They are generated automatically when you run a simulation.
