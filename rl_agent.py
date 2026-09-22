"""
rl_agent.py
-----------
A dependency-light Q-learning agent using linear function approximation
(no PyTorch/TensorFlow needed). One weight vector per action:

    Q(s, a) = w_a . s

This is intentionally simple: the state space here is a handful of
continuous circuit-features (see environment.py), so a linear
approximator with epsilon-greedy exploration and TD(0) updates is
sufficient to learn a decent pass-ordering policy, and it trains in
seconds on a laptop CPU.
"""

from __future__ import annotations
import numpy as np
from environment import CircuitEnv, N_ACTIONS, STOP_ACTION, ACTION_MENU


class QLearningAgent:
    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS,
                 lr: float = 0.05, gamma: float = 0.95,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay_episodes: int = 250, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_actions, state_dim))
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.n_actions = n_actions
        self.rng = rng

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state

    def epsilon(self, episode: int) -> float:
        frac = min(1.0, episode / max(1, self.eps_decay_episodes))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: np.ndarray, episode: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon(episode):
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def update(self, state, action, reward, next_state, done):
        q_sa = self.W[action] @ state
        target = reward if done else reward + self.gamma * np.max(self.q_values(next_state))
        td_error = target - q_sa
        self.W[action] += self.lr * td_error * state


def train(agent: QLearningAgent, circuits: dict, episodes: int = 300, verbose=True):
    """Train `agent` on a dict of {name: QuantumCircuit}, cycling through
    them each episode."""
    names = list(circuits.keys())
    history = []
    for ep in range(episodes):
        name = names[ep % len(names)]
        env = CircuitEnv(circuits[name])
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, ep)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
        history.append(ep_reward)
        if verbose and (ep + 1) % max(1, episodes // 10) == 0:
            avg = np.mean(history[-max(1, episodes // 10):])
            print(f"  [RL] episode {ep+1}/{episodes}  avg reward (last batch)={avg:.3f}  eps={agent.epsilon(ep):.2f}")
    return history


def optimize_with_policy(agent: QLearningAgent, circuit, max_steps: int = 12):
    """Run the greedy learned policy once on `circuit`, return optimized circuit + trace."""
    env = CircuitEnv(circuit, max_steps=max_steps)
    state = env.reset()
    trace = []
    done = False
    while not done:
        action = agent.act(state, episode=10**9, greedy=True)  # fully greedy
        state, reward, done, info = env.step(action)
        trace.append((ACTION_MENU[action][0], reward))
        if action == STOP_ACTION:
            break
    return env.result(), trace
