
import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT = "ablation_v9_summary.csv"
OUT = "ablation_v9_analysis"

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip()

required = [
    "Circuit", "Method", "Depth_mean",
    "Gates_mean", "Runtime_s_mean",
    "Estimated_Fidelity_mean"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}\nAvailable: {df.columns.tolist()}"
    )

df.to_csv(
    os.path.join(OUT, "per_circuit_method_summary.csv"),
    index=False
)

metrics = [
    "Depth_mean", "Gates_mean", "1Q_Gates_mean",
    "2Q_Gates_mean", "Runtime_s_mean",
    "Estimated_Fidelity_mean", "Process_Fidelity_mean",
    "Steps_mean", "Improved_Actions_mean",
    "Candidate_Calls_mean"
]
metrics = [c for c in metrics if c in df.columns]

overall = df.groupby("Method", as_index=False)[metrics].mean()
overall.to_csv(
    os.path.join(OUT, "overall_method_summary.csv"),
    index=False
)

plots = {
    "Depth_mean": ("Circuit depth comparison", "Mean depth"),
    "Gates_mean": ("Gate count comparison", "Mean gates"),
    "Runtime_s_mean": ("Runtime comparison", "Runtime (seconds)"),
    "Estimated_Fidelity_mean": (
        "Estimated fidelity comparison", "Estimated fidelity"
    ),
    "Process_Fidelity_mean": (
        "Process fidelity comparison", "Process fidelity"
    ),
    "Improved_Actions_mean": (
        "Improved actions comparison", "Mean improved actions"
    ),
    "Steps_mean": ("Optimization steps comparison", "Mean steps"),
}

for metric, (title, ylabel) in plots.items():
    if metric not in df.columns:
        continue

    pivot = df.pivot_table(
        index="Circuit",
        columns="Method",
        values=metric,
        aggfunc="mean"
    )

    if pivot.empty or pivot.notna().sum().sum() == 0:
        continue

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Circuit")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    filename = metric.replace("_mean", "").lower() + "_comparison.png"
    plt.savefig(
        os.path.join(OUT, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("\nAnalysis completed successfully.")
print("Output folder:", os.path.abspath(OUT))
print("\nOverall method averages:")
print(overall.round(4).to_string(index=False))
print("\nGenerated files:")
for filename in sorted(os.listdir(OUT)):
    print(" -", filename)
