"""Tests for Grid World RL Agent.

Students: DO NOT modify this file. These tests define the expected behavior
of your implementations. Run them with: pytest tests/test_grid_world.py
"""

import pytest
from lib.grid_world import GridWorld, ValueIteration, QLearning, SARSA, PolicyVisualizer


# --- Shared Fixtures ---

def make_simple_grid():
    """4x4 grid with 1 goal, 1 trap, 1 wall.

    Layout:
    . . . G
    . # . X
    . . . .
    S . . .

    S=(3,0) start, G=(0,3) goal, X=(1,3) trap, #=(1,1) wall
    """
    return GridWorld(
        rows=4, cols=4,
        walls=[(1, 1)],
        traps=[(1, 3)],
        goals=[(0, 3)],
        goal_reward=1.0,
        trap_reward=-1.0,
        step_reward=-0.04,
    )


def follow_policy(env, policy, start, max_steps=20):
    """Follow a policy from start state and return list of states visited."""
    states = [start]
    state = start
    for _ in range(max_steps):
        if env.is_terminal(state):
            break
        if state not in policy:
            break
        action = policy[state]
        state, _, done = env.step(state, action)
        states.append(state)
        if done:
            break
    return states


# === TestGridWorld (5 tests) ===

class TestGridWorld:
    """Tests for the pre-built GridWorld environment."""

    def test_states_count(self):
        """4x4 grid with 1 wall should have 15 states."""
        env = make_simple_grid()
        states = env.get_states()
        assert len(states) == 15

    def test_wall_not_in_states(self):
        """Wall cell (1,1) should not appear in states."""
        env = make_simple_grid()
        states = env.get_states()
        assert (1, 1) not in states

    def test_terminal_states(self):
        """Goal and trap should be terminal; other states should not."""
        env = make_simple_grid()
        assert env.is_terminal((0, 3)) is True   # goal
        assert env.is_terminal((1, 3)) is True   # trap
        assert env.is_terminal((0, 0)) is False
        assert env.is_terminal((2, 2)) is False

    def test_step_into_wall(self):
        """Stepping into a wall should keep agent in place."""
        env = make_simple_grid()
        # From (0,1), move DOWN would go to (1,1) which is a wall
        next_state, reward, done = env.step((0, 1), 2)  # DOWN
        assert next_state == (0, 1)
        assert done is False

    def test_step_off_grid(self):
        """Stepping off the grid should keep agent in place."""
        env = make_simple_grid()
        # From (0,0), move UP would go off grid
        next_state, reward, done = env.step((0, 0), 0)  # UP
        assert next_state == (0, 0)
        assert done is False


# === TestValueIteration (7 tests) ===

class TestValueIteration:
    """Tests for the ValueIteration algorithm."""

    def test_solve_returns_self(self):
        """solve() should return self for method chaining."""
        env = make_simple_grid()
        vi = ValueIteration(env, gamma=0.99, theta=1e-6)
        result = vi.solve()
        assert result is vi

    def test_policy_has_all_states(self):
        """Policy should have an action for every non-terminal state."""
        env = make_simple_grid()
        vi = ValueIteration(env).solve()
        policy = vi.get_policy()
        non_terminal = [s for s in env.get_states() if not env.is_terminal(s)]
        for state in non_terminal:
            assert state in policy, f"State {state} missing from policy"
            assert policy[state] in GridWorld.ACTIONS

    def test_values_has_all_states(self):
        """Values should exist for every state (including terminals)."""
        env = make_simple_grid()
        vi = ValueIteration(env).solve()
        values = vi.get_values()
        for state in env.get_states():
            assert state in values, f"State {state} missing from values"

    def test_goal_neighbors_point_to_goal(self):
        """States adjacent to goal should have policy pointing toward it."""
        env = make_simple_grid()
        vi = ValueIteration(env, gamma=0.99).solve()
        policy = vi.get_policy()
        # (0,2) is left of goal (0,3) — should go RIGHT (action 1)
        assert policy[(0, 2)] == 1, "State (0,2) should point RIGHT toward goal"

    def test_goal_value_highest(self):
        """Goal state should have the highest value."""
        env = make_simple_grid()
        vi = ValueIteration(env).solve()
        values = vi.get_values()
        goal_value = values[(0, 3)]
        for state, value in values.items():
            if state != (0, 3):
                assert goal_value >= value, (
                    f"Goal value {goal_value} should be >= {value} at {state}"
                )

    def test_trap_value_negative(self):
        """Trap state should have a negative value."""
        env = make_simple_grid()
        vi = ValueIteration(env).solve()
        values = vi.get_values()
        assert values[(1, 3)] < 0, "Trap value should be negative"

    def test_solves_simple_grid(self):
        """Following the optimal policy from (3,0) should reach the goal."""
        env = make_simple_grid()
        vi = ValueIteration(env, gamma=0.99).solve()
        policy = vi.get_policy()
        path = follow_policy(env, policy, (3, 0), max_steps=20)
        assert (0, 3) in path, (
            f"Policy should lead to goal (0,3). Path taken: {path}"
        )


# === TestQLearning (7 tests) ===

class TestQLearning:
    """Tests for the Q-Learning algorithm."""

    def test_train_returns_episode_rewards(self):
        """train() should return dict with 'episode_rewards' key."""
        env = make_simple_grid()
        ql = QLearning(env, random_seed=42)
        result = ql.train(n_episodes=100)
        assert isinstance(result, dict)
        assert "episode_rewards" in result

    def test_episode_rewards_length(self):
        """episode_rewards should have length equal to n_episodes."""
        env = make_simple_grid()
        ql = QLearning(env, random_seed=42)
        result = ql.train(n_episodes=200)
        assert len(result["episode_rewards"]) == 200

    def test_q_table_populated(self):
        """Q-table should have entries after training."""
        env = make_simple_grid()
        ql = QLearning(env, random_seed=42)
        ql.train(n_episodes=500)
        q_values = ql.get_q_values()
        assert len(q_values) > 0
        # At least some Q-values should be non-zero
        non_zero = [v for v in q_values.values() if v != 0.0]
        assert len(non_zero) > 0, "Some Q-values should be non-zero after training"

    def test_policy_has_states(self):
        """Policy should cover non-terminal states visited during training."""
        env = make_simple_grid()
        ql = QLearning(env, random_seed=42)
        ql.train(n_episodes=1000)
        policy = ql.get_policy()
        # At least the start state should be in the policy
        assert (3, 0) in policy
        # Policy actions should be valid
        for state, action in policy.items():
            assert action in GridWorld.ACTIONS

    def test_rewards_improve(self):
        """Mean reward should improve from first 100 to last 100 episodes."""
        env = make_simple_grid()
        ql = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        result = ql.train(n_episodes=3000, max_steps=100)
        rewards = result["episode_rewards"]
        first_100_mean = sum(rewards[:100]) / 100
        last_100_mean = sum(rewards[-100:]) / 100
        assert last_100_mean > first_100_mean, (
            f"Rewards should improve: first 100 mean={first_100_mean:.3f}, "
            f"last 100 mean={last_100_mean:.3f}"
        )

    def test_learns_good_policy(self):
        """After sufficient training, policy near goal should point toward it."""
        env = make_simple_grid()
        ql = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        ql.train(n_episodes=5000, max_steps=100)
        policy = ql.get_policy()
        # (0,2) is left of goal — should learn to go RIGHT (action 1)
        assert policy.get((0, 2)) == 1, (
            f"State (0,2) should point RIGHT toward goal, got action "
            f"{policy.get((0, 2))}"
        )

    def test_deterministic(self):
        """Same seed should produce same Q-values."""
        env = make_simple_grid()
        ql1 = QLearning(env, random_seed=123)
        ql1.train(n_episodes=500)
        ql2 = QLearning(env, random_seed=123)
        ql2.train(n_episodes=500)
        q1 = ql1.get_q_values()
        q2 = ql2.get_q_values()
        for key in q1:
            assert abs(q1[key] - q2[key]) < 1e-10, (
                f"Q-values differ at {key}: {q1[key]} vs {q2[key]}"
            )


# === TestSARSA (6 tests) ===

class TestSARSA:
    """Tests for the SARSA algorithm."""

    def test_train_returns_episode_rewards(self):
        """train() should return dict with 'episode_rewards' key."""
        env = make_simple_grid()
        sarsa = SARSA(env, random_seed=42)
        result = sarsa.train(n_episodes=100)
        assert isinstance(result, dict)
        assert "episode_rewards" in result

    def test_episode_rewards_length(self):
        """episode_rewards should have length equal to n_episodes."""
        env = make_simple_grid()
        sarsa = SARSA(env, random_seed=42)
        result = sarsa.train(n_episodes=300)
        assert len(result["episode_rewards"]) == 300

    def test_policy_has_states(self):
        """Policy should cover visited states."""
        env = make_simple_grid()
        sarsa = SARSA(env, random_seed=42)
        sarsa.train(n_episodes=1000)
        policy = sarsa.get_policy()
        assert (3, 0) in policy
        for state, action in policy.items():
            assert action in GridWorld.ACTIONS

    def test_rewards_improve(self):
        """Mean reward should improve from first 100 to last 100 episodes."""
        env = make_simple_grid()
        sarsa = SARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        result = sarsa.train(n_episodes=3000, max_steps=100)
        rewards = result["episode_rewards"]
        first_100_mean = sum(rewards[:100]) / 100
        last_100_mean = sum(rewards[-100:]) / 100
        assert last_100_mean > first_100_mean, (
            f"Rewards should improve: first 100 mean={first_100_mean:.3f}, "
            f"last 100 mean={last_100_mean:.3f}"
        )

    def test_learns_goal_direction(self):
        """After training, policy near goal should point toward it."""
        env = make_simple_grid()
        sarsa = SARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        sarsa.train(n_episodes=5000, max_steps=100)
        policy = sarsa.get_policy()
        # (0,2) is left of goal — should learn to go RIGHT
        assert policy.get((0, 2)) == 1, (
            f"State (0,2) should point RIGHT toward goal, got action "
            f"{policy.get((0, 2))}"
        )

    def test_deterministic(self):
        """Same seed should produce same results."""
        env = make_simple_grid()
        s1 = SARSA(env, random_seed=99)
        s1.train(n_episodes=500)
        s2 = SARSA(env, random_seed=99)
        s2.train(n_episodes=500)
        q1 = s1.get_q_values()
        q2 = s2.get_q_values()
        for key in q1:
            assert abs(q1[key] - q2[key]) < 1e-10, (
                f"Q-values differ at {key}: {q1[key]} vs {q2[key]}"
            )


# === TestPolicyVisualizer (4 tests) ===

class TestPolicyVisualizer:
    """Tests for the PolicyVisualizer."""

    def test_render_policy_contains_arrows(self):
        """Rendered policy should contain arrow characters."""
        env = make_simple_grid()
        viz = PolicyVisualizer(env)
        # Create a simple policy
        policy = {}
        for state in env.get_states():
            if not env.is_terminal(state):
                policy[state] = 1  # all RIGHT
        output = viz.render_policy(policy)
        arrows = set("↑→↓←")
        found = any(ch in output for ch in arrows)
        assert found, f"Expected arrows in output, got:\n{output}"

    def test_render_policy_goal_and_trap(self):
        """Rendered policy should show G for goal and X for trap."""
        env = make_simple_grid()
        viz = PolicyVisualizer(env)
        policy = {s: 0 for s in env.get_states() if not env.is_terminal(s)}
        output = viz.render_policy(policy)
        assert "G" in output, f"Expected 'G' for goal in output:\n{output}"
        assert "X" in output, f"Expected 'X' for trap in output:\n{output}"

    def test_render_policy_wall(self):
        """Rendered policy should show # for walls."""
        env = make_simple_grid()
        viz = PolicyVisualizer(env)
        policy = {s: 0 for s in env.get_states() if not env.is_terminal(s)}
        output = viz.render_policy(policy)
        assert "#" in output, f"Expected '#' for wall in output:\n{output}"

    def test_render_values_format(self):
        """Rendered values should contain decimal numbers."""
        env = make_simple_grid()
        viz = PolicyVisualizer(env)
        values = {s: 0.5 for s in env.get_states()}
        output = viz.render_values(values)
        assert "0.50" in output, f"Expected '0.50' in output:\n{output}"
        assert "#" in output, f"Expected '#' for wall in output:\n{output}"


# === TestComparison (3 tests) ===

class TestComparison:
    """Tests comparing all three algorithms."""

    def test_all_methods_find_goal(self):
        """All three methods should find a policy that reaches the goal."""
        env = make_simple_grid()

        # Value Iteration — has full knowledge, test from far corner
        vi = ValueIteration(env, gamma=0.99).solve()
        vi_path = follow_policy(env, vi.get_policy(), (3, 0), max_steps=20)
        assert (0, 3) in vi_path, f"ValueIteration failed to reach goal: {vi_path}"

        # Q-Learning — trains from (0,0), test from (0,0)
        ql = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        ql.train(n_episodes=5000, max_steps=100)
        ql_path = follow_policy(env, ql.get_policy(), (0, 0), max_steps=20)
        assert (0, 3) in ql_path, f"QLearning failed to reach goal: {ql_path}"

        # SARSA — trains from (0,0), test from (0,0)
        sarsa = SARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        sarsa.train(n_episodes=5000, max_steps=100)
        sarsa_path = follow_policy(env, sarsa.get_policy(), (0, 0), max_steps=20)
        assert (0, 3) in sarsa_path, f"SARSA failed to reach goal: {sarsa_path}"

    def test_value_iteration_optimal(self):
        """Value iteration should find an optimal (shortest) path."""
        env = make_simple_grid()
        vi = ValueIteration(env, gamma=0.99).solve()
        vi_path = follow_policy(env, vi.get_policy(), (3, 0), max_steps=20)
        # Optimal path from (3,0) to (0,3) on this grid is 6 steps
        # (3,0)->(2,0)->(1,0)->(0,0)->(0,1)->(0,2)->(0,3) = 7 states = 6 steps
        assert len(vi_path) <= 8, (
            f"Value iteration path too long ({len(vi_path)} states): {vi_path}"
        )

    def test_q_vs_sarsa_both_learn(self):
        """Both Q-Learning and SARSA should show reward improvement."""
        env = make_simple_grid()

        ql = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        ql_result = ql.train(n_episodes=3000, max_steps=100)
        ql_rewards = ql_result["episode_rewards"]
        ql_improvement = (
            sum(ql_rewards[-100:]) / 100 - sum(ql_rewards[:100]) / 100
        )

        sarsa = SARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1, random_seed=42)
        sarsa_result = sarsa.train(n_episodes=3000, max_steps=100)
        sarsa_rewards = sarsa_result["episode_rewards"]
        sarsa_improvement = (
            sum(sarsa_rewards[-100:]) / 100 - sum(sarsa_rewards[:100]) / 100
        )

        assert ql_improvement > 0, (
            f"Q-Learning should improve over training, got {ql_improvement:.3f}"
        )
        assert sarsa_improvement > 0, (
            f"SARSA should improve over training, got {sarsa_improvement:.3f}"
        )
