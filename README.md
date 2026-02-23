# Grid World RL Agent

A ZeroCourse project for Course 14.3: Reinforcement Learning (Week 1).

## What You'll Build

Implement reinforcement learning algorithms to solve a configurable grid world environment. The grid has walls, traps, and goals. Your agent must learn to navigate from a start position to the goal while avoiding traps.

You'll implement three different algorithms and a visualizer:

| Class | Methods | Description |
|-------|---------|-------------|
| `ValueIteration` | `__init__`, `solve`, `get_policy`, `get_values` | Dynamic programming — computes optimal policy with full environment knowledge |
| `QLearning` | `__init__`, `_epsilon_greedy`, `train`, `get_policy`, `get_q_values` | Off-policy TD control — learns by exploring with epsilon-greedy |
| `SARSA` | `__init__`, `_epsilon_greedy`, `train`, `get_policy`, `get_q_values` | On-policy TD control — learns from actions actually taken |
| `PolicyVisualizer` | `__init__`, `render_policy`, `render_values` | Text-based visualization of policies and value functions |

The `GridWorld` environment is already implemented for you.

## The Grid World

```
. . . G      Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
. # . X      G = Goal (+1.0 reward)
. . . .      X = Trap (-1.0 reward)
S . . .      # = Wall (can't enter)
              . = Empty (-0.04 per step)
```

- Moving into a wall or off the grid keeps you in place
- Goal and trap are terminal states (episode ends)
- The small negative step reward encourages finding the goal quickly

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests (most will fail initially):
   ```bash
   pytest tests/
   ```

3. Open `lib/grid_world.py` and implement each class.

4. Run the tests again to check your progress.

## Tips

- **Start with `ValueIteration`** — it's pure dynamic programming with no randomness. Initialize V(s) = 0 for all states, then repeatedly update each state's value using the Bellman equation: V(s) = max_a [R(s,a) + gamma * V(s')].

- **Then implement `PolicyVisualizer`** — being able to see your policies makes debugging the TD methods much easier.

- **`QLearning` next** — the key update rule is: Q(s,a) += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a)). Use `random.Random(seed)` for reproducible epsilon-greedy exploration.

- **`SARSA` last** — it's very similar to Q-Learning. The only difference: instead of using max_a' Q(s',a'), you use Q(s',a') where a' is the action you actually chose with epsilon-greedy. This makes it on-policy.

- For epsilon-greedy: with probability epsilon pick a random action, otherwise pick the action with the highest Q-value.

- Use `env.step(state, action)` to interact with the environment. It returns `(next_state, reward, done)`.

- Use `env.get_states()` to get all valid states and `env.is_terminal(state)` to check for terminal states.

- Episodes start at state `(0, 0)`.

## Running Tests

```bash
pytest tests/                              # Run all tests
pytest tests/ -v                           # Verbose output
pytest tests/ -k "TestValueIteration"      # Run only ValueIteration tests
pytest tests/ -k "TestQLearning"           # Run only QLearning tests
pytest tests/ -k "TestSARSA"              # Run only SARSA tests
pytest tests/ -k "TestPolicyVisualizer"    # Run only visualizer tests
pytest tests/ -k "TestComparison"          # Run only comparison tests
```
