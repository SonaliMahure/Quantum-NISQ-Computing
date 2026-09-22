"""
environment.py
--------------
A lightweight Gym-style environment in which an agent (RL or GA) chooses,
step by step, which Qiskit transpiler pass to apply next to a circuit.

State  : a feature vector describing the *current* circuit.
Action : index into a fixed menu of Qiskit TransformationPass objects,
         plus a STOP action.
Reward : reduction in a weighted "NISQ cost" (depth + gate count + a
         gate-error-based fidelity penalty) caused by the chosen pass.

The environment never changes the logical function of the circuit -
every pass in the menu is a correctness-preserving Qiskit transpiler
pass, so the *unitary* of the circuit is provably unchanged. What
changes is depth / gate count, which we use as a cheap, standard proxy
for expected fidelity on NISQ hardware (see `estimate_fidelity` below).
"""

from __future__ import annotations
import numpy as np

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGatesDecomposition,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    Collect2qBlocks,
    ConsolidateBlocks,
    RemoveDiagonalGatesBeforeMeasure,
    InverseCancellation,
)
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate, CZGate

# --------------------------------------------------------------------------
# Realistic-ish per-gate error rates for a superconducting NISQ device.
# Used only to turn (gate count, depth) into a single scalar "estimated
# fidelity" for the reward signal - NOT a physical simulation.
# --------------------------------------------------------------------------
ERR_1Q = 3e-4
ERR_2Q = 1.0e-2


def estimate_fidelity(circuit: QuantumCircuit) -> float:
    """Depolarizing-style estimate of circuit fidelity from gate counts."""
    n1q, n2q = _gate_split(circuit)
    return ((1 - ERR_1Q) ** n1q) * ((1 - ERR_2Q) ** n2q)


def _gate_split(circuit: QuantumCircuit):
    n1q, n2q = 0, 0
    for instr in circuit.data:
        nq = instr.operation.num_qubits
        if instr.operation.name in ("barrier", "measure"):
            continue
        if nq == 1:
            n1q += 1
        elif nq >= 2:
            n2q += 1
    return n1q, n2q


def circuit_cost(circuit: QuantumCircuit, w_depth=1.0, w_gates=0.3, w_fid=50.0) -> float:
    """Lower is better. Combines depth, total gate count, and fidelity loss."""
    depth = circuit.depth()
    n_gates = sum(1 for i in circuit.data if i.operation.name not in ("barrier", "measure"))
    fid = estimate_fidelity(circuit)
    return w_depth * depth + w_gates * n_gates + w_fid * (1 - fid)


# --------------------------------------------------------------------------
# Action menu: each entry returns a fresh PassManager containing one
# (or a short, coupled sequence of) Qiskit transformation pass(es).
# --------------------------------------------------------------------------
def _inverse_pairs():
    return [(HGate(), HGate()), (XGate(), XGate()), (YGate(), YGate()),
            (ZGate(), ZGate()), (CXGate(), CXGate()), (CZGate(), CZGate())]


def build_action_menu():
    return [
        ("optimize_1q", lambda: PassManager([Optimize1qGatesDecomposition()])),
        ("commutative_cancellation", lambda: PassManager([CommutativeCancellation()])),
        ("cx_cancellation", lambda: PassManager([ CommutativeCancellation()])),
        ("commutative_inverse_cancellation", lambda: PassManager([CommutativeInverseCancellation()])),
        ("consolidate_blocks", lambda: PassManager([Collect2qBlocks(), ConsolidateBlocks()])),
        ("remove_diag_before_measure", lambda: PassManager([RemoveDiagonalGatesBeforeMeasure()])),
        ("inverse_cancellation", lambda: PassManager([InverseCancellation(_inverse_pairs())])),
        ("stop", None),  # terminal action
    ]


ACTION_MENU = build_action_menu()
N_ACTIONS = len(ACTION_MENU)
STOP_ACTION = N_ACTIONS - 1


class CircuitEnv:
    """One episode = optimizing a single circuit via a sequence of passes."""

    def __init__(self, circuit: QuantumCircuit, max_steps: int = 12,
                 patience: int = 3):
        self.original = circuit.copy()
        self.max_steps = max_steps
        self.patience = patience
        self.reset()

    def reset(self):
        self.circuit = self.original.copy()
        self.steps = 0
        self.no_improve_streak = 0
        self.prev_cost = circuit_cost(self.circuit)
        return self._state()

    def _state(self) -> np.ndarray:
        depth = self.circuit.depth()
        n1q, n2q = _gate_split(self.circuit)
        total = n1q + n2q
        fid = estimate_fidelity(self.circuit)
        # normalize roughly to O(1) magnitudes for the linear Q-function
        return np.array([
            depth / 50.0,
            total / 100.0,
            n2q / 50.0,
            n1q / 100.0,
            fid,
            self.steps / self.max_steps,
            self.no_improve_streak / self.patience,
        ], dtype=np.float64)

    def step(self, action: int):
        """Returns (next_state, reward, done, info)."""
        self.steps += 1
        name, factory = ACTION_MENU[action]

        if action == STOP_ACTION:
            return self._state(), 0.0, True, {"pass": "stop"}

        pm: PassManager = factory()
        try:
            new_circuit = pm.run(self.circuit)
        except Exception as e:
            # Pass not applicable (e.g. no matching gates) -> no-op, small penalty
            return self._state(), -0.05, self._maybe_done(), {"pass": name, "error": str(e)}

        new_cost = circuit_cost(new_circuit)
        reward = self.prev_cost - new_cost  # positive if circuit improved
        improved = reward > 1e-9

        self.circuit = new_circuit
        self.prev_cost = new_cost
        self.no_improve_streak = 0 if improved else self.no_improve_streak + 1

        done = self._maybe_done()
        return self._state(), reward, done, {"pass": name, "improved": improved}

    def _maybe_done(self) -> bool:
        return (self.steps >= self.max_steps) or (self.no_improve_streak >= self.patience)

    def result(self) -> QuantumCircuit:
        return self.circuit
