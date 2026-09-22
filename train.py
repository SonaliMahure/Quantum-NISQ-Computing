"""
train.py
--------
End-to-end driver:
  1. Build benchmark circuits.
  2. Train the RL (Q-learning) agent across all benchmark circuits.
  3. Run the GA optimizer independently per circuit.
  4. Transpile each circuit with Qiskit's built-in optimization_level 0-3.
  5. Print a side-by-side comparison table per circuit.

Usage:
    python train.py --episodes 300 --generations 40 --circuits ghz5 qft4 bv6
"""
import argparse
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import circuit

from environment import CircuitEnv
from rl_agent import QLearningAgent, train as train_rl, optimize_with_policy
from genetic_optimizer import GeneticOptimizer
from benchmarks import (
    default_benchmark_suite, evaluate_circuit, baseline_qiskit_levels,
    print_comparison_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300,
                        help="RL training episodes")
    parser.add_argument("--generations", type=int, default=40,
                        help="GA generations per circuit")
    parser.add_argument("--pop-size", type=int, default=40,
                        help="GA population size")
    parser.add_argument("--circuits", nargs="*", default=None,
                        help="Subset of benchmark circuit names to use (default: all)")
    args = parser.parse_args()

    suite = default_benchmark_suite()

    if args.circuits:
        suite = {
            k: v for k, v in suite.items()
            if k in args.circuits
        }

        if not suite:
            raise SystemExit(
                f"No matching circuits. Available: {list(default_benchmark_suite())}"
            )

    print(f"Benchmark circuits: {list(suite.keys())}\n")

    # ---------------------------------------------------------
    # 1. Train RL agent
    # ---------------------------------------------------------
    state_dim = len(
        CircuitEnv(next(iter(suite.values()))).reset()
    )

    agent = QLearningAgent(state_dim=state_dim)

    print("Training RL agent...")

    training_history = train_rl(
        agent,
        suite,
        episodes=args.episodes
    )

    print()

    # ---------------------------------------------------------
    # 2. Save RL training history
    # ---------------------------------------------------------
    training_df = pd.DataFrame({
        "Episode": range(1, len(training_history) + 1),
        "Reward": training_history,
        "Epsilon": [
            agent.epsilon(ep)
            for ep in range(len(training_history))
        ]
    })

    training_df["Moving_Average"] = (
        training_df["Reward"]
        .rolling(window=20)
        .mean()
    )

    training_df.to_csv(
        "RL_training_history.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 3. Generate RL training curve
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        training_df["Episode"],
        training_df["Reward"],
        alpha=0.35,
        label="Episode Reward"
    )

    plt.plot(
        training_df["Episode"],
        training_df["Moving_Average"],
        linewidth=2,
        label="20-Episode Moving Average"
    )

    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.title("Q-Learning Training Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "RL_training_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("RL training table saved as RL_training_history.csv")
    print("RL training curve saved as RL_training_curve.png")
    print()

    # ---------------------------------------------------------
    # 4. Evaluate each benchmark circuit
    # ---------------------------------------------------------
    for name, circuit in suite.items():

        original_gate_count = sum(
            1
            for i in circuit.data
            if i.operation.name not in ("barrier", "measure")
        )

        print(
            f"=== {name} "
            f"(qubits={circuit.num_qubits}, "
            f"orig depth={circuit.depth()}, "
            f"orig gates={original_gate_count}) ==="
        )

        rows = []

        # -----------------------------------------------------
        # RL
        # -----------------------------------------------------
        print("\n===== BEFORE RL =====")
        print("Depth      :", circuit.depth())
        print("Gate Count :", len(circuit.data))
        print("Operations :", circuit.count_ops())

        t0 = time.perf_counter()

        rl_circuit, trace = optimize_with_policy(
            agent,
            circuit
        )

        rl_time = time.perf_counter() - t0

        print("\n===== AFTER RL =====")
        print("Depth      :", rl_circuit.depth())
        print("Gate Count :", len(rl_circuit.data))
        print("Operations :", rl_circuit.count_ops())

        rows.append(
            evaluate_circuit(
                "RL (Q-learning)",
                circuit,
                rl_circuit,
                rl_time
            )
        )

        # -----------------------------------------------------
        # GA
        # -----------------------------------------------------
        ga = GeneticOptimizer(
            pop_size=args.pop_size
        )

        t0 = time.perf_counter()

        ga_circuit, best_chrom, ga_history = ga.run(
            circuit,
            generations=args.generations,
            verbose=False
        )

        ga_time = time.perf_counter() - t0

        rows.append(
            evaluate_circuit(
                "GA",
                circuit,
                ga_circuit,
                ga_time
            )
        )

        # -----------------------------------------------------
        # Qiskit baselines
        # -----------------------------------------------------
        baseline_results = baseline_qiskit_levels(
            circuit
        )

        rows.extend(
            baseline_results.values()
        )

        print_comparison_table(rows)
        print()


if __name__ == "__main__":
    main()
