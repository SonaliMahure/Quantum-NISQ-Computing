
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project directory: assumes this script is saved beside the CSV.
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ablation_v9_results.csv"
OUTPUT_DIR = BASE_DIR / "ablation_analysis"

OUTPUT_DIR.mkdir(exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. "
        "Check the CSV filename and project folder."
    )

df = pd.read_csv(INPUT_FILE)

# Verify the essential columns.
required = [
    "Circuit", "Method", "Depth", "Gates",
    "Runtime_s", "Estimated_Fidelity",
    "Process_Fidelity", "Correct", "Steps",
    "Candidate_Calls", "Changed_Actions",
    "Improved_Actions"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Convert numeric metrics safely.
numeric_cols = [
    "Depth", "Gates", "1Q_Gates", "2Q_Gates",
    "Estimated_Fidelity", "Process_Fidelity",
    "Runtime_s", "Steps", "Candidate_Calls",
    "Changed_Actions", "Improved_Actions",
    "Timeouts", "Timeout_Retries", "Worker_Errors"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert correctness values to numeric success indicators.
correct_text = df["Correct"].astype(str).str.strip().str.lower()
df["Correct_Rate"] = correct_text.map({
    "true": 1.0, "false": 0.0,
    "1": 1.0, "0": 0.0,
    "yes": 1.0, "no": 0.0
})

# 1. Summary across all circuits and seeds, by method.
metrics = [
    c for c in [
        "Depth", "Gates", "1Q_Gates", "2Q_Gates",
        "Estimated_Fidelity", "Process_Fidelity",
        "Correct_Rate", "Runtime_s", "Steps",
        "Candidate_Calls", "Changed_Actions",
        "Improved_Actions", "Timeouts",
        "Timeout_Retries", "Worker_Errors"
    ] if c in df.columns
]

summary = df.groupby("Method")[metrics].agg(["mean", "std"])
summary.columns = [
    f"{metric}_{stat}" for metric, stat in summary.columns
]
summary = summary.reset_index()
summary.to_csv(
    OUTPUT_DIR / "ablation_summary_by_method.csv",
    index=False
)

# 2. Circuit-level summary, grouped by method.
per_circuit = df.groupby(
    ["Circuit", "Method"]
)[metrics].mean().reset_index()

per_circuit.to_csv(
    OUTPUT_DIR / "ablation_summary_by_circuit_method.csv",
    index=False
)

# 3. Preserve a clean copy of the input data for analysis.
df.to_csv(OUTPUT_DIR / "ablation_data_for_plotting.csv", index=False)

# 4. Publication-quality plots.
# Plot means per method; separate circuits are shown where useful.
plot_specs = [
    ("Depth", "Mean circuit depth", "mean_depth.png"),
    ("Gates", "Mean total gate count", "mean_gates.png"),
    ("Runtime_s", "Mean runtime (seconds)", "mean_runtime.png"),
    ("Estimated_Fidelity", "Mean estimated fidelity",
     "mean_estimated_fidelity.png"),
    ("Process_Fidelity", "Mean process fidelity",
     "mean_process_fidelity.png"),
    ("Improved_Actions", "Mean improved actions",
     "mean_improved_actions.png"),
    ("Steps", "Mean optimization steps", "mean_steps.png"),
]

for metric, ylabel, filename in plot_specs:
    if metric not in df.columns:
        continue

    # Show each circuit separately, averaged across its seeds.
    plot_data = (
        df.groupby(["Circuit", "Method"])[metric]
        .mean()
        .reset_index()
    )

    pivot = plot_data.pivot(
        index="Circuit", columns="Method", values=metric
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 5),
        width=0.8
    )
    ax.set_title(ylabel + " by circuit and method")
    ax.set_xlabel("Benchmark circuit")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Method")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

print("\nAnalysis completed successfully.")
print(f"Input rows: {len(df)}")
print(f"Methods: {df['Method'].nunique()}")
print(f"Circuits: {df['Circuit'].nunique()}")
print(f"Seeds: {df['Seed'].nunique() if 'Seed' in df.columns else 'N/A'}")
print(f"Output folder: {OUTPUT_DIR}")
print("\nSummary by method:")
print(summary.round(4).to_string(index=False))
