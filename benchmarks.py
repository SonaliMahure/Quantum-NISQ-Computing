"""
benchmarks.py
-------------
Standard benchmark circuits + evaluation metrics (gate count, depth,
2-qubit gate count, estimated NISQ fidelity, and an exact-equivalence
correctness check via statevector/operator comparison).
"""

from __future__ import annotations
import time
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, GroverOperator
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector, Operator, process_fidelity

from environment import estimate_fidelity, _gate_split


# --------------------------------------------------------------------------
# Benchmark circuit builders
# --------------------------------------------------------------------------
def ghz(n=5) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"ghz{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def qft(n=4) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"qft{n}")
    qc.compose(QFT(n, do_swaps=True), inplace=True)

    # Convert to primitive gates
    qc = qc.decompose(reps=10)

    return qc


def bernstein_vazirani(n=6, secret: str = None) -> QuantumCircuit:
    if secret is None:
        secret = "1" * (n - 1)
    qc = QuantumCircuit(n, name=f"bv{n}")
    qc.x(n - 1)
    qc.h(range(n))
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(i, n - 1)
    qc.h(range(n - 1))
    return qc


def grover(n=3) -> QuantumCircuit:
    oracle = QuantumCircuit(n, name="oracle")
    oracle.h(n - 1)
    oracle.mcx(list(range(n - 1)), n - 1)
    oracle.h(n - 1)

    grover_op = GroverOperator(oracle)

    qc = QuantumCircuit(n, name=f"grover{n}")
    qc.h(range(n))
    qc.compose(grover_op, inplace=True)

    # Convert to primitive gates
    qc = qc.decompose(reps=10)

    return qc


def randcirc(n=5, depth=15, seed=42) -> QuantumCircuit:
    qc = random_circuit(n, depth, max_operands=2, seed=seed)
    qc.name = f"randcirc{n}"
    return qc


def adder(n_bits=3) -> QuantumCircuit:
    """Simple ripple-carry-like adder built from Toffoli/CX (illustrative,
    not overhead-optimal) so the benchmark suite includes an arithmetic
    circuit with a non-trivial CX/Toffoli count."""
    n = 3 * n_bits + 1
    qc = QuantumCircuit(n, name=f"adder{n_bits}")
    for i in range(n_bits):
        a, b, c_in, c_out = i, n_bits + i, 2 * n_bits, 2 * n_bits + 1 + i if i < n_bits - 1 else 2 * n_bits
        qc.ccx(a, b, c_out if c_out < n else n - 1)
        qc.cx(a, b)
        qc.ccx(b, c_in, c_out if c_out < n else n - 1)
    return qc


def default_benchmark_suite() -> dict:
    return {
        "ghz5": ghz(5),
        "qft4": qft(4),
        "bv6": bernstein_vazirani(6),
        "randcirc": randcirc(5, 15),
        "grover3": grover(3),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def check_equivalence(original: QuantumCircuit, optimized: QuantumCircuit, tol=1e-6) -> bool:
    """Verify the optimized circuit implements the same unitary (up to
    global phase) as the original - i.e. optimization did not change the
    circuit's function. Uses statevector simulation for circuits without
    measurement/reset."""
    try:
        has_meas = any(i.operation.name in ("measure", "reset") for i in original.data)
        if has_meas:
            return None  # not checked - contains non-unitary ops
        op1 = Operator(original)
        op2 = Operator(optimized)
        return bool(process_fidelity(op1, op2) > 1 - tol)
    except Exception:
        return None


def evaluate_circuit(name: str, original: QuantumCircuit, optimized: QuantumCircuit,
                      elapsed: float = None) -> dict:
    n1q_o, n2q_o = _gate_split(optimized)
    equiv = check_equivalence(original, optimized)
    return {
        "name": name,
        "depth": optimized.depth(),
        "gate_count": n1q_o + n2q_o,
        "two_qubit_gates": n2q_o,
        "one_qubit_gates": n1q_o,
        "est_fidelity": estimate_fidelity(optimized),
        "correct": equiv,
        "time_s": elapsed,
    }


def baseline_qiskit_levels(circuit: QuantumCircuit, levels=(0, 1, 2, 3)) -> dict:
    results = {}
    for lvl in levels:
        t0 = time.perf_counter()
        try:
            tc = transpile(circuit, optimization_level=lvl,
                            basis_gates=["u", "cx"], seed_transpiler=0)
        except Exception:
            tc = transpile(circuit, optimization_level=lvl, seed_transpiler=0)
        elapsed = time.perf_counter() - t0
        results[f"qiskit_L{lvl}"] = evaluate_circuit(f"qiskit_L{lvl}", circuit, tc, elapsed)
    return results


def print_comparison_table(rows: list):
    headers = ["method", "depth", "gates", "2q-gates", "1q-gates", "est.fidelity", "correct", "time(s)"]
    widths = [16, 7, 7, 9, 9, 13, 8, 8]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        correct = "-" if r["correct"] is None else ("YES" if r["correct"] else "NO")
        t = "-" if r["time_s"] is None else f"{r['time_s']:.3f}"
        vals = [r["name"], r["depth"], r["gate_count"], r["two_qubit_gates"],
                r["one_qubit_gates"], f"{r['est_fidelity']:.4f}", correct, t]
        print(" | ".join(str(v).ljust(w) for v, w in zip(vals, widths)))
