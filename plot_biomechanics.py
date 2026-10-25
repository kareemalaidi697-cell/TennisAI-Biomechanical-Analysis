
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 1. Load real measurements from the tennis video
csv_path = Path("runs/tennis_joint_angles.csv")
output_dir = Path("runs/biomechanics_plots")
output_dir.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

df = pd.read_csv(csv_path)

time = pd.to_numeric(df["time_seconds"], errors="coerce")
left_knee = pd.to_numeric(df["left_knee_deg"], errors="coerce")
right_knee = pd.to_numeric(df["right_knee_deg"], errors="coerce")

# Keep missing measurements as gaps.
# Remove only impossible values; do not invent or clip measurements.
left_knee = left_knee.where(left_knee.between(0, 180))
right_knee = right_knee.where(right_knee.between(0, 180))

# 2. Create the figure
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

# 3. Plot the actual knee measurements
ax.plot(
    time, left_knee,
    label="Left Knee",
    color="#1f77b4",
    linewidth=2
)

ax.plot(
    time, right_knee,
    label="Right Knee",
    color="#ff7f0e",
    linewidth=2
)

# 4. Configure axes
ax.set_title("Tennis Biomechanics - Knees", fontsize=14)
ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_ylabel("2D Joint Angle (degrees)", fontsize=12)

ax.set_xlim(0, max(42, time.max()))
ax.set_ylim(0, 185)

ax.set_xticks(range(0, int(max(42, time.max())) + 1, 5))
ax.set_yticks(range(0, 181, 20))

ax.tick_params(axis="both", which="major", labelsize=13)

# 5. Grid and legend
ax.grid(True, linestyle="-", linewidth=0.5, color="#d9d9d9")
ax.set_axisbelow(True)
ax.legend(loc="lower left", fontsize=14, frameon=True)

plt.tight_layout()

# 6. Save the graph
output_file = output_dir / "knees_angles_real.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("\nGraph saved to:", output_file.resolve())
print("Total frames:", len(df))
print("Valid left-knee measurements:", left_knee.notna().sum())
print("Valid right-knee measurements:", right_knee.notna().sum())

