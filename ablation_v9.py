"""

Experimental methods
---------------------
1. Full RL          : full circuit_cost improvement
2. Depth-only RL    : depth improvement
3. Gate-only RL     : total gate-count improvement
4. Random action    : uniform random unique transformation

All methods use the same action set, step budget, timeout policy, seeds and
benchmarks.  Only the action-selection policy/objective differs.

Timeout
-------
Each expensive pass is isolated in a Windows child process:
5 s first attempt -> fresh 30 s retry from the original pre-action circuit.
No action can wait indefinitely.

Outputs
-------
ablation_v9_results.csv
ablation_v9_summary.csv
ablation_v9_action_trace.csv
ablation_v9_policy_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit.quantum_info import Operator, process_fidelity

from environment import (
    CircuitEnv,
    ACTION_MENU,
    STOP_ACTION,
    N_ACTIONS,
    circuit_cost,
    estimate_fidelity,
    _gate_split,
)
from benchmarks import default_benchmark_suite


FAST_DEFAULT = 5.0
SLOW_DEFAULT = 30.0
EPS = 1e-10

# Corrected unique action menu.
# action 2 ("cx_cancellation") is an exact duplicate of action 1 in
# environment.py, so it is excluded from the experimental action set.
UNIQUE_ENV_ACTIONS = [0, 1, 3, 4, 5, 6]


def action_name(action: int) -> str:
    item = ACTION_MENU[int(action)]
    return str(item[0]) if isinstance(item, (tuple, list)) else str(item)


def unique_action_names() -> list[str]:
    return [action_name(a) for a in UNIQUE_ENV_ACTIONS]


def metrics(circuit):
    n1, n2 = _gate_split(circuit)
    return {
        "depth": int(circuit.depth()),
        "gates": int(n1 + n2),
        "1q": int(n1),
        "2q": int(n2),
        "fidelity": float(estimate_fidelity(circuit)),
        "cost": float(circuit_cost(circuit)),
    }


def objective_reward(before, after, mode):
    mb = metrics(before)
    ma = metrics(after)

    if mode == "full":
        return float(mb["cost"] - ma["cost"])
    if mode == "depth":
        return float(mb["depth"] - ma["depth"])
    if mode == "gates":
        return float(mb["gates"] - ma["gates"])

    raise ValueError(f"Unknown objective mode: {mode}")


def normalized_reward(raw, before_cost):
    return float(
        np.clip(raw / max(abs(float(before_cost)), 1.0), -1.0, 1.0)
    )


def context_key(circuit, last_action, streak):
    """
    Compact, deterministic context.

    Exact circuit dimensions are intentionally coarse-binned so that the
    policy learns reusable behavior rather than memorizing every transient
    cost value.
    """
    m = metrics(circuit)
    return (
        min(m["depth"] // 5, 20),
        min(m["gates"] // 10, 20),
        min(m["2q"] // 5, 20),
        int(last_action if last_action is not None else -1),
        min(int(streak), 3),
    )


@dataclass
class Stat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x):
        self.n += 1
        delta = float(x) - self.mean
        self.mean += delta / self.n
        delta2 = float(x) - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        if self.n < 2:
            return 0.25
        return max(self.m2 / (self.n - 1), 1e-8)


class ContextualBanditPolicy:
    """
    Conservative empirical contextual bandit.

    A state-context statistic is shrunk toward the action's global mean.
    UCB exploration is applied only while the action has insufficient
    evidence. Repeating a non-improving action immediately is discouraged.
    """

    def __init__(self, actions, alpha=0.25, seed=0):
        self.actions = list(actions)
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)

        self.global_stats = {a: Stat() for a in self.actions}
        self.context_stats = {}

    def _stats(self, context, action):
        key = (context, action)
        if key not in self.context_stats:
            self.context_stats[key] = Stat()
        return self.context_stats[key]

    def update(self, context, action, reward):
        self.global_stats[action].update(reward)
        self._stats(context, action).update(reward)

    def score(self, context, action, total_observations, last_action):
        g = self.global_stats[action]
        c = self._stats(context, action)

        if c.n:
            weight = c.n / (c.n + 4.0)
            mean = weight * c.mean + (1.0 - weight) * g.mean
            uncertainty = np.sqrt(c.variance / max(c.n, 1))
        else:
            mean = g.mean
            uncertainty = np.sqrt(g.variance)

        # Force each unique action to be sampled early.
        if g.n < 2:
            exploration = 0.75
        else:
            exploration = 0.20 * np.sqrt(
                np.log(total_observations + 2.0) / g.n
            )

        repeat_penalty = 0.0
        if last_action == action and g.n >= 2:
            repeat_penalty = 0.10

        return mean + self.alpha * uncertainty + exploration - repeat_penalty

    def choose(self, context, epsilon, total_observations, last_action):
        if self.rng.random() < epsilon:
            candidates = [
                a for a in self.actions if a != last_action
            ] or self.actions
            return int(self.rng.choice(candidates))

        scores = np.array(
            [
                self.score(
                    context,
                    a,
                    total_observations,
                    last_action,
                )
                for a in self.actions
            ]
        )

        best = np.flatnonzero(
            np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12)
        )
        return int(self.actions[int(self.rng.choice(best))])

    def rank_actions(self, context, total_observations, last_action=None, epsilon=0.0):
        """Return all actions in learned preference order.

        During exploration, use a randomized permutation. During exploitation,
        rank by the contextual/UCB score. A recent action is moved to the end
        when alternatives exist to reduce immediate repeats.
        """
        actions = list(self.actions)
        if epsilon > 0.0 and self.rng.random() < float(epsilon):
            self.rng.shuffle(actions)
            return actions

        scored = [(self.score(context, a, total_observations, last_action), a)
                  for a in actions]
        scored.sort(key=lambda x: (-x[0], x[1]))
        ranked = [a for _, a in scored]
        if last_action in ranked and len(ranked) > 1:
            ranked.remove(last_action)
            ranked.append(last_action)
        return ranked

    def epsilon(self, episode, total_episodes):
        if total_episodes <= 1:
            return 0.10
        frac = min(1.0, episode / float(total_episodes - 1))
        return 0.20 - 0.15 * frac


def worker(circuit, action, max_steps, patience, out_q):
    try:
        env = CircuitEnv(
            circuit.copy(),
            max_steps=int(max_steps),
            patience=int(patience),
        )
        env.reset()
        state, env_reward, done, info = env.step(int(action))

        out_q.put(
            (
                "ok",
                env.circuit.copy(),
                np.asarray(state, dtype=float),
                float(env_reward),
                bool(done),
                dict(info) if isinstance(info, dict) else str(info),
            ),
            block=True,
            timeout=2.0,
        )
    except BaseException as exc:
        try:
            out_q.put(
                ("error", type(exc).__name__, str(exc)),
                block=True,
                timeout=1.0,
            )
        except Exception:
            pass


def terminate_worker(p):
    if not p.is_alive():
        p.join(0.1)
        return

    try:
        p.terminate()
    except Exception:
        pass

    p.join(1.0)

    if p.is_alive():
        try:
            p.kill()
        except Exception:
            pass
        p.join(1.0)


def isolated_step(circuit, action, args, timeout_s):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=1)

    p = ctx.Process(
        target=worker,
        args=(
            circuit.copy(),
            int(action),
            args.max_steps,
            args.patience,
            out_q,
        ),
        daemon=True,
    )

    t0 = time.perf_counter()

    try:
        p.start()
        p.join(float(timeout_s))
    except BaseException:
        terminate_worker(p)
        try:
            out_q.close()
            out_q.cancel_join_thread()
        except Exception:
            pass
        raise

    elapsed = time.perf_counter() - t0

    if p.is_alive():
        terminate_worker(p)
        try:
            out_q.close()
            out_q.cancel_join_thread()
        except Exception:
            pass
        return None, "timeout", elapsed

    result = None
    try:
        result = out_q.get(timeout=0.5)
    except (queue.Empty, EOFError, OSError):
        pass
    finally:
        try:
            out_q.close()
            out_q.cancel_join_thread()
        except Exception:
            pass

    if result is None:
        return None, f"worker_exit_{p.exitcode}", elapsed

    if result[0] != "ok":
        return result, "error", elapsed

    return result, "ok", elapsed


def safe_step(circuit, action, args):
    result, status, elapsed = isolated_step(
        circuit, action, args, args.qiskit_timeout
    )

    retried = False

    if status == "timeout":
        retried = True
        result2, status2, elapsed2 = isolated_step(
            circuit, action, args, args.slow_qiskit_timeout
        )
        result = result2
        status = status2
        elapsed += elapsed2

    return result, status, elapsed, retried


def equivalence_fidelity(original, candidate):
    """Process fidelity used only for an improving candidate."""
    try:
        return float(process_fidelity(Operator(original), Operator(candidate)))
    except Exception:
        return np.nan


def evaluate_candidates(current, objective, policy, args, rng, last_action,
                        total_observations, epsilon, phase, circuit_name,
                        method_label, seed, episode, step, trace):
    """Evaluate a bounded prefix of a policy ranking and accept only an improvement.

    This is the central V9 correction: a non-improving candidate is NEVER made
    the new current state. The candidate budget is identical for learned and
    random policies, making the ablation comparable in Qiskit-call budget.
    """
    context = context_key(current, last_action, 0)
    ranked = policy.rank_actions(context, total_observations, last_action, epsilon)
    ranked = ranked[:max(1, int(args.candidate_budget))]

    best_candidate = current.copy()
    best_m = metrics(current)
    best_action = None
    best_raw = 0.0
    best_fid = 1.0
    rows = []
    total_runtime = 0.0
    retries = 0
    timeouts = 0
    errors = 0

    for rank, action in enumerate(ranked, start=1):
        before = current.copy()
        bm = metrics(before)
        result, status, runtime, retried = safe_step(before, action, args)
        total_runtime += runtime
        retries += int(retried)
        timeouts += int(status == "timeout")
        errors += int(status not in ("ok", "timeout"))

        if status == "ok":
            after = result[1]
            am = metrics(after)
            raw = objective_reward(before, after, objective)
            changed = (
                bm["depth"] != am["depth"] or
                bm["gates"] != am["gates"] or
                abs(bm["cost"] - am["cost"]) > EPS
            )
            improved = raw > EPS
            candidate_pf = np.nan
            accepted = False
            reject_reason = "not_improving"

            if improved:
                candidate_pf = equivalence_fidelity(before, after)
                if np.isnan(candidate_pf) or candidate_pf >= args.fidelity_threshold:
                    accepted = True
                    reject_reason = "accepted"
                    if (best_action is None or raw > best_raw + EPS):
                        best_candidate = after.copy()
                        best_m = am
                        best_action = action
                        best_raw = raw
                        best_fid = candidate_pf if not np.isnan(candidate_pf) else 1.0
                else:
                    reject_reason = "fidelity_rejected"
        else:
            after = before.copy()
            am = bm
            raw = -args.failure_penalty
            changed = False
            improved = False
            candidate_pf = np.nan
            accepted = False
            reject_reason = status

        policy_reward = normalized_reward(raw if status == "ok" else -args.failure_penalty,
                                          bm["cost"])
        policy.update(context, action, policy_reward)
        rows.append({
            "Phase": phase, "Circuit": circuit_name, "Method": method_label,
            "Seed": seed, "Episode": episode, "Step": step,
            "Candidate_Rank": rank, "Action": action,
            "Action_Name": action_name(action), "Status": status,
            "Retried_5_to_30": retried, "Before_Cost": bm["cost"],
            "After_Cost": am["cost"], "Raw_Reward": raw,
            "Normalized_Reward": policy_reward, "Improved": improved,
            "Accepted": accepted, "Reject_Reason": reject_reason,
            "Candidate_Process_Fidelity": candidate_pf,
            "Runtime_s": runtime, "Epsilon": epsilon,
        })

        # For a deterministic greedy search, once the first ranked action
        # improves we need not pay for lower-ranked candidates. Exploration
        # still permits a bounded number of candidates when the first fails.
        if accepted and args.first_improvement_stop:
            break

    return (best_candidate, best_m, best_action, best_raw, best_fid,
            rows, total_runtime, retries, timeouts, errors)


def train_policy(suite, objective, args, seed, trace):
    policy = ContextualBanditPolicy(
        UNIQUE_ENV_ACTIONS, alpha=args.exploration, seed=seed
    )
    names = list(suite)
    total_observations = 0

    for ep in range(args.episodes):
        name = names[ep % len(names)]
        current = suite[name].copy()
        last_action = None
        no_improve = 0
        epsilon = policy.epsilon(ep, args.episodes)

        print(f"      Train {ep + 1}/{args.episodes} | {objective} | seed={seed} | {name}", flush=True)
        for step in range(1, args.max_steps + 1):
            before_best = metrics(current)
            (candidate, cm, chosen, reward, _, rows, runtime, retries,
             timeouts, errors) = evaluate_candidates(
                current, objective, policy, args, policy.rng, last_action,
                total_observations, epsilon, "train", name, objective,
                seed, ep + 1, step, trace
            )
            trace.extend(rows)
            total_observations += len(rows)

            if chosen is not None and cm["cost"] < before_best["cost"] - EPS:
                current = candidate
                last_action = chosen
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= args.patience:
                break
    return policy


def evaluate_policy(suite, objective, label, policy, args, seed, results, trace):
    for name, original in suite.items():
        current = original.copy()
        best = current.copy()
        best_m = metrics(best)
        last_action = None
        no_improve = 0
        runtime_total = 0.0
        retries = timeouts = errors = changed = improved_count = candidate_calls = 0

        print(f"    [EVAL] {label} | seed={seed} | {name}", flush=True)
        for step in range(1, args.max_steps + 1):
            before_best = metrics(current)
            (candidate, cm, chosen, reward, _, rows, runtime, rr,
             tt, ee) = evaluate_candidates(
                current, objective, policy, args, np.random.default_rng(seed + step),
                last_action, 10**9, 0.0, "eval", name, label, seed,
                "eval", step, trace
            )
            trace.extend(rows)
            runtime_total += runtime
            retries += rr; timeouts += tt; errors += ee; candidate_calls += len(rows)
            changed += sum(int(r["Status"] == "ok" and r["Before_Cost"] != r["After_Cost"]) for r in rows)
            improved_count += sum(int(r["Improved"]) for r in rows)

            if chosen is not None and cm["cost"] < before_best["cost"] - EPS:
                current = candidate.copy()
                last_action = chosen
                no_improve = 0
                if cm["cost"] < best_m["cost"] - EPS:
                    best = candidate.copy(); best_m = dict(cm)
            else:
                no_improve += 1

            if no_improve >= args.patience:
                break

        pf = equivalence_fidelity(original, best)
        correct = bool(not np.isnan(pf) and pf >= args.fidelity_threshold)
        results.append({
            "Circuit": name, "Method": label, "Seed": seed,
            "Depth": best_m["depth"], "Gates": best_m["gates"],
            "1Q_Gates": best_m["1q"], "2Q_Gates": best_m["2q"],
            "Estimated_Fidelity": best_m["fidelity"],
            "Process_Fidelity": pf, "Correct": correct,
            "Runtime_s": runtime_total, "Steps": step,
            "Candidate_Calls": candidate_calls,
            "Changed_Actions": changed, "Improved_Actions": improved_count,
            "Timeouts": timeouts, "Timeout_Retries": retries,
            "Worker_Errors": errors,
        })


def evaluate_random(suite, args, seed, results, trace):
    rng = np.random.default_rng(100000 + seed)
    for name, original in suite.items():
        current = original.copy(); best = current.copy(); best_m = metrics(best)
        no_improve = 0; runtime_total = 0.0; retries = timeouts = errors = 0
        changed = improved_count = candidate_calls = 0
        print(f"    [RANDOM] seed={seed} | {name}", flush=True)

        for step in range(1, args.max_steps + 1):
            actions = list(UNIQUE_ENV_ACTIONS); rng.shuffle(actions)
            actions = actions[:max(1, int(args.candidate_budget))]
            before_best = metrics(current)
            rows = []
            accepted = None; accepted_circuit = None; accepted_m = None; accepted_raw = 0.0
            for rank, action in enumerate(actions, start=1):
                before = current.copy(); bm = metrics(before)
                result, status, runtime, retried = safe_step(before, action, args)
                runtime_total += runtime; retries += int(retried)
                timeouts += int(status == "timeout"); errors += int(status not in ("ok", "timeout")); candidate_calls += 1
                if status == "ok":
                    after = result[1]; am = metrics(after); raw = bm["cost"] - am["cost"]
                    did_change = abs(bm["cost"] - am["cost"]) > EPS or bm["depth"] != am["depth"] or bm["gates"] != am["gates"]
                    improved = raw > EPS
                    pf = equivalence_fidelity(before, after) if improved else np.nan
                    ok = improved and (np.isnan(pf) or pf >= args.fidelity_threshold)
                    reason = "accepted" if ok else ("fidelity_rejected" if improved else "not_improving")
                    rows.append({"Phase":"eval","Circuit":name,"Method":"Random action","Seed":seed,"Episode":"eval","Step":step,"Candidate_Rank":rank,"Action":action,"Action_Name":action_name(action),"Status":status,"Retried_5_to_30":retried,"Before_Cost":bm["cost"],"After_Cost":am["cost"],"Raw_Reward":raw,"Normalized_Reward":normalized_reward(raw,bm["cost"]),"Improved":improved,"Accepted":ok,"Reject_Reason":reason,"Candidate_Process_Fidelity":pf,"Runtime_s":runtime,"Epsilon":0.0})
                    changed += int(did_change); improved_count += int(improved)
                    if ok:
                        accepted = action; accepted_circuit = after.copy(); accepted_m = am; accepted_raw = raw
                        if args.first_improvement_stop: break
                else:
                    rows.append({"Phase":"eval","Circuit":name,"Method":"Random action","Seed":seed,"Episode":"eval","Step":step,"Candidate_Rank":rank,"Action":action,"Action_Name":action_name(action),"Status":status,"Retried_5_to_30":retried,"Before_Cost":bm["cost"],"After_Cost":bm["cost"],"Raw_Reward":-args.failure_penalty,"Normalized_Reward":-args.failure_penalty,"Improved":False,"Accepted":False,"Reject_Reason":status,"Candidate_Process_Fidelity":np.nan,"Runtime_s":runtime,"Epsilon":0.0})
            trace.extend(rows)
            if accepted is not None and accepted_m["cost"] < before_best["cost"] - EPS:
                current = accepted_circuit; no_improve = 0
                if accepted_m["cost"] < best_m["cost"] - EPS:
                    best = current.copy(); best_m = dict(accepted_m)
            else: no_improve += 1
            if no_improve >= args.patience: break

        pf = equivalence_fidelity(original, best); correct = bool(not np.isnan(pf) and pf >= args.fidelity_threshold)
        results.append({"Circuit":name,"Method":"Random action","Seed":seed,"Depth":best_m["depth"],"Gates":best_m["gates"],"1Q_Gates":best_m["1q"],"2Q_Gates":best_m["2q"],"Estimated_Fidelity":best_m["fidelity"],"Process_Fidelity":pf,"Correct":correct,"Runtime_s":runtime_total,"Steps":step,"Candidate_Calls":candidate_calls,"Changed_Actions":changed,"Improved_Actions":improved_count,"Timeouts":timeouts,"Timeout_Retries":retries,"Worker_Errors":errors})


def diagnostic(args):
    suite = default_benchmark_suite(args.qasm_dir)

    if args.circuits:
        wanted = set(args.circuits)
        suite = {
            k: v for k, v in suite.items()
            if k in wanted
        }

    rows = []

    print("=" * 78)
    print("V9 ACTION / CANDIDATE DIAGNOSTIC")
    print("Six UNIQUE transformation actions from environment.py")
    print("Duplicate environment action 2 is intentionally excluded.")
    print("=" * 78, flush=True)

    for name, circuit in suite.items():
        bm = metrics(circuit)

        print(
            f"\n{name}: depth={bm['depth']} gates={bm['gates']}",
            flush=True,
        )

        for action in UNIQUE_ENV_ACTIONS:
            result, status, runtime, retried = safe_step(
                circuit,
                action,
                args,
            )

            row = {
                "Circuit": name,
                "Action": action,
                "Action_Name": action_name(action),
                "Status": status,
                "Runtime_s": runtime,
                "Retried_5_to_30": retried,
            }

            if status == "ok":
                am = metrics(result[1])
                raw = bm["cost"] - am["cost"]
                row.update(
                    {
                        "Depth_before": bm["depth"],
                        "Depth_after": am["depth"],
                        "Gates_before": bm["gates"],
                        "Gates_after": am["gates"],
                        "Cost_before": bm["cost"],
                        "Cost_after": am["cost"],
                        "Improvement": raw,
                        "Improved": raw > EPS,
                    }
                )

            rows.append(row)

    diagnostic_output = getattr(
        args,
        "diagnostic_output",
        "ablation_v9_diagnostic.csv",
    )
    pd.DataFrame(rows).to_csv(
        diagnostic_output,
        index=False,
    )

    print(
        f"\nCreated: {diagnostic_output}",
        flush=True,
    )


def write_policy_report(results, trace, path):
    rdf = pd.DataFrame(results)
    tdf = pd.DataFrame(trace)

    lines = [    ]

    if not rdf.empty:
        summary = (
            rdf.groupby("Method")
            .agg(
                mean_depth=("Depth", "mean"),
                mean_gates=("Gates", "mean"),
                mean_runtime_s=("Runtime_s", "mean"),
                mean_steps=("Steps", "mean"),
                mean_improvements=("Improved_Actions", "mean"),
                mean_retries=("Timeout_Retries", "mean"),
                mean_pf=("Process_Fidelity", "mean"),
            )
            .reset_index()
        )

        lines.append("Evaluation summary:")
        lines.append(summary.to_string(index=False))
        lines.append("")

    if not tdf.empty:
        ev = tdf[tdf["Phase"] == "eval"].copy()
        if not ev.empty:
            counts = (
                ev.groupby(["Method", "Action_Name"])
                .size()
                .reset_index(name="Selections")
            )
            lines.append("Evaluation action selections:")
            lines.append(counts.to_string(index=False))
            lines.append("")

    lines.extend(
        [
            "Interpretation rule:",
            "  This should only be called an effective policy if it improves",
            "  the final objective relative to Random without sacrificing",
            "  process fidelity/correctness. Otherwise the result is reported",
            "  as a negative/neutral ablation rather than tuned until it wins.",
        ]
    )

    Path(path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    mp.freeze_support()

    p = argparse.ArgumentParser(
        description="Ablation"
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--qiskit-timeout", type=float, default=FAST_DEFAULT)
    p.add_argument("--slow-qiskit-timeout", type=float, default=SLOW_DEFAULT)
    p.add_argument("--exploration", type=float, default=0.25)
    p.add_argument("--failure-penalty", type=float, default=0.05)
    p.add_argument("--candidate-budget", type=int, default=2,
                   help="maximum Qiskit candidates evaluated per policy step")
    p.add_argument("--first-improvement-stop", action="store_true", default=True,
                   help="stop candidate search after first valid improvement")
    p.add_argument("--no-first-improvement-stop", dest="first_improvement_stop",
                   action="store_false")
    p.add_argument("--fidelity-threshold", type=float, default=1.0 - 1e-8)
    p.add_argument("--circuits", nargs="*", default=None)
    p.add_argument(
    "--qasm-dir",
    type=str,
    default=None,
    help="Optional directory containing OpenQASM 2.0 files",
)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--diagnostic-output", default="ablation_v9_diagnostic.csv")
    p.add_argument("--results-output", default="ablation_v9_results.csv")
    p.add_argument("--summary-output", default="ablation_v9_summary.csv")
    p.add_argument("--trace-output", default="ablation_v9_action_trace.csv")
    p.add_argument(
        "--policy-report",
        default="ablation_v9_policy_report.txt",
    )
    args = p.parse_args()

    if args.slow_qiskit_timeout < args.qiskit_timeout:
        raise SystemExit(
            "--slow-qiskit-timeout must be >= --qiskit-timeout"
        )

    print("=" * 78)
    print("Starting ablation_v9.py")
    print(
        f"Timeout: {args.qiskit_timeout:.1f}s -> "
        f"{args.slow_qiskit_timeout:.1f}s"
    )
    print(
        f"episodes={args.episodes}, seeds={args.seeds}, "
        f"max_steps={args.max_steps}, patience={args.patience}"
    )
    print(
        "Unique actions: "
        + ", ".join(unique_action_names())
    )
    print(
        "STOP: excluded from learned optimization policy."
    )
    print("=" * 78, flush=True)

    if args.diagnostic:
        diagnostic(args)
        return

    suite = default_benchmark_suite(args.qasm_dir)

    if args.circuits:
        wanted = set(args.circuits)
        suite = {
            k: v for k, v in suite.items()
            if k in wanted
        }

    if not suite:
        raise SystemExit("No matching benchmark circuits.")

    print(
        f"Loaded {len(suite)} benchmark circuit(s): "
        + ", ".join(suite.keys()),
        flush=True,
    )

    results = []
    trace = []

    methods = [
        ("full", "Full RL"),
        ("depth", "Depth-only RL"),
        ("gates", "Gate-only RL"),
    ]

    for objective, label in methods:
        for seed in range(args.seeds):
            print(
                f"\n{'=' * 78}\n"
                f"TRAINING {label} | seed={seed}\n"
                f"{'=' * 78}",
                flush=True,
            )

            policy = train_policy(
                suite,
                objective,
                args,
                seed,
                trace,
            )

            evaluate_policy(
                suite,
                objective,
                label,
                policy,
                args,
                seed,
                results,
                trace,
            )

    for seed in range(args.seeds):
        print(
            f"\n{'=' * 78}\n"
            f"RANDOM BASELINE | seed={seed}\n"
            f"{'=' * 78}",
            flush=True,
        )

        evaluate_random(
            suite,
            args,
            seed,
            results,
            trace,
        )

    rdf = pd.DataFrame(results)
    rdf.to_csv(args.results_output, index=False)

    numeric = [
        "Depth",
        "Gates",
        "1Q_Gates",
        "2Q_Gates",
        "Estimated_Fidelity",
        "Process_Fidelity",
        "Runtime_s",
        "Steps",
        "Changed_Actions",
        "Improved_Actions",
        "Timeouts",
        "Timeout_Retries",
        "Worker_Errors",
        "Candidate_Calls",
    ]

    summary_rows = []

    for (circuit, method), g in rdf.groupby(
        ["Circuit", "Method"]
    ):
        row = {
            "Circuit": circuit,
            "Method": method,
        }

        for col in numeric:
            vals = pd.to_numeric(
                g[col],
                errors="coerce",
            )
            row[col + "_mean"] = float(vals.mean())
            row[col + "_std"] = float(vals.std())

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.summary_output, index=False)

    tdf = pd.DataFrame(trace)
    tdf.to_csv(args.trace_output, index=False)

    write_policy_report(
        results,
        trace,
        args.policy_report,
    )

    print("\n" + "=" * 78)
    print("V9 ABLATION COMPLETED")
    print("=" * 78)
    print(f"Created: {args.results_output}")
    print(f"Created: {args.summary_output}")
    print(f"Created: {args.trace_output}")
    print(f"Created: {args.policy_report}")
    print("\nMean results:")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
