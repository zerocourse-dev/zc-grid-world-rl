# Grid World RL Agent
#
# Implement the ValueIteration, QLearning, SARSA, and PolicyVisualizer classes below.
# The GridWorld environment is already implemented for you.
#
# Run `pytest` to check your solutions.
#
# Hint: Start with ValueIteration (it's the simplest — pure dynamic programming).
# Then implement PolicyVisualizer so you can see your policies.
# Then tackle QLearning, and finally SARSA (very similar to Q-learning).

import random


class GridWorld:
    """Configurable grid world environment.

    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    Cell types: 'empty', 'wall', 'trap', 'goal'

    Example:
        >>> env = GridWorld(rows=4, cols=4, walls=[(1,1)], traps=[(1,3)], goals=[(0,3)])
        >>> state, reward, done = env.step((3, 0), 1)  # move RIGHT from (3,0)
        >>> state
        (3, 1)
    """

    ACTIONS = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT
    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
    DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, rows=4, cols=4, walls=None, traps=None, goals=None,
                 goal_reward=1.0, trap_reward=-1.0, step_reward=-0.04):
        self.rows = rows
        self.cols = cols
        self.walls = set(walls or [])
        self.traps = set(traps or [])
        self.goals = set(goals or [])
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.step_reward = step_reward

    def get_states(self):
        """Return list of all non-wall states as (row, col) tuples.

        Example:
            >>> env = GridWorld(rows=2, cols=2, walls=[(0,1)])
            >>> env.get_states()
            [(0, 0), (1, 0), (1, 1)]
        """
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.walls]

    def is_terminal(self, state):
        """Check if state is a terminal state (goal or trap).

        Example:
            >>> env = GridWorld(goals=[(0,3)], traps=[(1,3)])
            >>> env.is_terminal((0, 3))
            True
        """
        return state in self.goals or state in self.traps

    def get_reward(self, state):
        """Get reward for entering a state.

        Example:
            >>> env = GridWorld(goals=[(0,3)], goal_reward=1.0, step_reward=-0.04)
            >>> env.get_reward((0, 3))
            1.0
            >>> env.get_reward((2, 2))
            -0.04
        """
        if state in self.goals:
            return self.goal_reward
        if state in self.traps:
            return self.trap_reward
        return self.step_reward

    def step(self, state, action):
        """Take an action, return (next_state, reward, done).

        If action would move into wall or off grid, stay in place.

        Example:
            >>> env = GridWorld(rows=4, cols=4, walls=[(1,1)], goals=[(0,3)])
            >>> env.step((0, 0), 0)  # UP from top-left -> stays
            ((0, 0), -0.04, False)
        """
        if self.is_terminal(state):
            return state, 0, True
        dr, dc = self.DELTAS[action]
        next_state = (state[0] + dr, state[1] + dc)
        if (next_state[0] < 0 or next_state[0] >= self.rows or
                next_state[1] < 0 or next_state[1] >= self.cols or
                next_state in self.walls):
            next_state = state
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def get_transition(self, state, action):
        """Deterministic transition: return (next_state, reward, done).

        Alias for step() — useful for planning algorithms.
        """
        return self.step(state, action)


class ValueIteration:
    """Value Iteration algorithm for solving MDPs.

    Computes optimal values and policy using dynamic programming.

    Example:
        >>> env = GridWorld(rows=4, cols=4, walls=[(1,1)], traps=[(1,3)], goals=[(0,3)])
        >>> vi = ValueIteration(env, gamma=0.99, theta=1e-6)
        >>> vi.solve()
        >>> vi.get_policy()[(0, 2)]  # state next to goal should point RIGHT
        1
        >>> vi.get_values()[(0, 3)]  # goal has highest value
        1.0

    @param env: GridWorld environment
    @param gamma: float, discount factor (default 0.99)
    @param theta: float, convergence threshold (default 1e-6)
    """

    def __init__(self, env, gamma=0.99, theta=1e-6):
        """Initialize value iteration with environment and parameters.

        @param env: GridWorld — the environment to solve
        @param gamma: float — discount factor (0 < gamma <= 1)
        @param theta: float — convergence threshold (stop when max change < theta)
        """
        raise NotImplementedError("Implement __init__")

    def solve(self, max_iterations=1000):
        """Run value iteration algorithm.

        Initialize V(s) = 0 for all non-terminal states. Set V(terminal) to
        the terminal reward. Repeat: for each non-terminal state, compute
        V(s) = max_a [R(s,a) + gamma * V(s')] where R(s,a) is the transition
        reward. For transitions into terminal states, the target is just the
        reward (no discounted future — the episode ends). Stop when
        max change < theta or max_iterations reached.

        Store results in self.values (dict: state -> value) and
        self.policy (dict: state -> best action).

        @param max_iterations: int — maximum number of iterations
        @return: self (for chaining)
        """
        raise NotImplementedError("Implement solve")

    def get_policy(self):
        """Return the computed policy.

        @return: dict mapping (row, col) -> action for each non-terminal state
        """
        raise NotImplementedError("Implement get_policy")

    def get_values(self):
        """Return the computed state values.

        @return: dict mapping (row, col) -> float value for each state
        """
        raise NotImplementedError("Implement get_values")


class QLearning:
    """Q-Learning algorithm (off-policy TD control).

    Learns Q-values through interaction with the environment using
    epsilon-greedy exploration.

    Example:
        >>> env = GridWorld(rows=4, cols=4, walls=[(1,1)], traps=[(1,3)], goals=[(0,3)])
        >>> ql = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        >>> result = ql.train(n_episodes=1000)
        >>> len(result['episode_rewards'])
        1000

    @param env: GridWorld environment
    @param alpha: float, learning rate (default 0.1)
    @param gamma: float, discount factor (default 0.99)
    @param epsilon: float, exploration rate (default 0.1)
    @param random_seed: int, seed for reproducibility (default 42)
    """

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42):
        """Initialize Q-Learning with environment and hyperparameters.

        Initialize Q-table as dict of (state, action) -> 0.0 for all
        state-action pairs. Use random.Random(random_seed) for the RNG.

        @param env: GridWorld — the environment
        @param alpha: float — learning rate
        @param gamma: float — discount factor
        @param epsilon: float — exploration rate for epsilon-greedy
        @param random_seed: int — random seed for reproducibility
        """
        raise NotImplementedError("Implement __init__")

    def _epsilon_greedy(self, state):
        """Choose action using epsilon-greedy strategy.

        With probability epsilon, pick a random action.
        Otherwise, pick the action with the highest Q-value.
        Break ties arbitrarily.

        @param state: tuple (row, col)
        @return: int — action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        raise NotImplementedError("Implement _epsilon_greedy")

    def train(self, n_episodes=1000, max_steps=100):
        """Train the agent using Q-Learning.

        For each episode: start at (0,0), take epsilon-greedy actions,
        update Q-values using:
            Q(s,a) += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))

        For terminal states, the target is just the reward (no future value).

        @param n_episodes: int — number of training episodes
        @param max_steps: int — max steps per episode
        @return: dict with 'episode_rewards' key (list of total reward per episode)
        """
        raise NotImplementedError("Implement train")

    def get_policy(self):
        """Extract greedy policy from Q-table.

        @return: dict mapping (row, col) -> best action (argmax Q)
        """
        raise NotImplementedError("Implement get_policy")

    def get_q_values(self):
        """Return the Q-table.

        @return: dict of (state, action) -> float Q-value
        """
        raise NotImplementedError("Implement get_q_values")


class SARSA:
    """SARSA algorithm (on-policy TD control).

    Like Q-Learning but updates using the action actually taken (on-policy),
    not the greedy action.

    Example:
        >>> env = GridWorld(rows=4, cols=4, walls=[(1,1)], traps=[(1,3)], goals=[(0,3)])
        >>> sarsa = SARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        >>> result = sarsa.train(n_episodes=1000)
        >>> len(result['episode_rewards'])
        1000

    @param env: GridWorld environment
    @param alpha: float, learning rate (default 0.1)
    @param gamma: float, discount factor (default 0.99)
    @param epsilon: float, exploration rate (default 0.1)
    @param random_seed: int, seed for reproducibility (default 42)
    """

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42):
        """Initialize SARSA with environment and hyperparameters.

        Initialize Q-table as dict of (state, action) -> 0.0 for all
        state-action pairs. Use random.Random(random_seed) for the RNG.

        @param env: GridWorld — the environment
        @param alpha: float — learning rate
        @param gamma: float — discount factor
        @param epsilon: float — exploration rate for epsilon-greedy
        @param random_seed: int — random seed for reproducibility
        """
        raise NotImplementedError("Implement __init__")

    def _epsilon_greedy(self, state):
        """Choose action using epsilon-greedy strategy.

        With probability epsilon, pick a random action.
        Otherwise, pick the action with the highest Q-value.
        Break ties arbitrarily.

        @param state: tuple (row, col)
        @return: int — action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        raise NotImplementedError("Implement _epsilon_greedy")

    def train(self, n_episodes=1000, max_steps=100):
        """Train the agent using SARSA.

        For each episode: start at (0,0), choose a using epsilon-greedy.
        Then loop: take action a, observe r, s'. Choose a' using epsilon-greedy.
        Update: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))
        For terminal s', the target is just r.

        @param n_episodes: int — number of training episodes
        @param max_steps: int — max steps per episode
        @return: dict with 'episode_rewards' key (list of total reward per episode)
        """
        raise NotImplementedError("Implement train")

    def get_policy(self):
        """Extract greedy policy from Q-table.

        @return: dict mapping (row, col) -> best action (argmax Q)
        """
        raise NotImplementedError("Implement get_policy")

    def get_q_values(self):
        """Return the Q-table.

        @return: dict of (state, action) -> float Q-value
        """
        raise NotImplementedError("Implement get_q_values")


class PolicyVisualizer:
    """Visualize policies and values on a grid world.

    Renders human-readable text representations of policies (with arrows)
    and value functions (with numbers).

    Example:
        >>> env = GridWorld(rows=2, cols=2, goals=[(0,1)])
        >>> viz = PolicyVisualizer(env)
        >>> policy = {(0, 0): 1, (1, 0): 0, (1, 1): 0}
        >>> print(viz.render_policy(policy))
        -> G
        ^  ^

    @param env: GridWorld environment
    """

    def __init__(self, env):
        """Initialize visualizer with environment.

        @param env: GridWorld — the environment to visualize
        """
        raise NotImplementedError("Implement __init__")

    def render_policy(self, policy):
        """Render policy as a grid with directional arrows.

        Use these Unicode symbols:
        - Actions: 0=UP -> '\\u2191', 1=RIGHT -> '\\u2192', 2=DOWN -> '\\u2193', 3=LEFT -> '\\u2190'
        - Goal: 'G'
        - Trap: 'X'
        - Wall: '#'
        - No policy entry: '.'

        Separate cells with at least one space. Separate rows with newlines.

        @param policy: dict mapping (row, col) -> action
        @return: str — multi-line grid visualization
        """
        raise NotImplementedError("Implement render_policy")

    def render_values(self, values):
        """Render state values as a grid with numbers.

        Format values to 2 decimal places. Use '#' for walls.
        Separate cells with spaces. Separate rows with newlines.

        @param values: dict mapping (row, col) -> float
        @return: str — multi-line grid visualization
        """
        raise NotImplementedError("Implement render_values")
