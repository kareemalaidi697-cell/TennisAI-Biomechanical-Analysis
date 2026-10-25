
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "runs"
    / "risk_analysis"
    / "tennis_motion_features.csv"
)

OUTPUT_DIR = BASE_DIR / "runs" / "risk_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# الأعمدة المطلوبة
required = [
    "time_seconds",
    "left_knee_deg",
    "right_knee_deg",
    "left_knee_deg_angular_speed_deg_s",
    "right_knee_deg_angular_speed_deg_s",
    "knee_angle_difference_deg",
    "review_flag",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# تحويل القيم إلى أرقام
for col in required:
    if col != "review_flag":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# تصحيح قيم الزوايا غير الصالحة
for col in ["left_knee_deg", "right_knee_deg"]:
    df.loc[~df[col].between(0, 180), col] = float("nan")

print("\n" + "=" * 55)
print("TENNIS AI - RISK RESULTS REVIEW")
print("=" * 55)

print(f"\nTotal frames: {len(df)}")

print("\nKnee angle difference:")
print(df["knee_angle_difference_deg"].describe())

print("\nAngular speed statistics (degrees/second):")

for col in [
    "left_knee_deg_angular_speed_deg_s",
    "right_knee_deg_angular_speed_deg_s",
]:
    print(f"\n{col}")
    print(df[col].abs().describe())

print("\nReview flags:")
print(df["review_flag"].value_counts(dropna=False))

# رسم زوايا الركبتين
plt.figure(figsize=(13, 5))

plt.plot(
    df["time_seconds"],
    df["left_knee_deg"],
    label="Left Knee",
    linewidth=1.5,
)

plt.plot(
    df["time_seconds"],
    df["right_knee_deg"],
    label="Right Knee",
    linewidth=1.5,
)

plt.xlabel("Time (seconds)")
plt.ylabel("2D Joint Angle (degrees)")
plt.title("Tennis Biomechanics - Knee Angles")
plt.ylim(0, 185)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

angle_path = OUTPUT_DIR / "knee_angles_review.png"
plt.savefig(angle_path, dpi=200)
plt.close()

# رسم سرعة تغير الزوايا
plt.figure(figsize=(13, 5))

plt.plot(
    df["time_seconds"],
    df["left_knee_deg_angular_speed_deg_s"],
    label="Left Knee Angular Speed",
    linewidth=1,
)

plt.plot(
    df["time_seconds"],
    df["right_knee_deg_angular_speed_deg_s"],
    label="Right Knee Angular Speed",
    linewidth=1,
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular Speed (degrees/second)")
plt.title("Knee Angular Speed Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

speed_path = OUTPUT_DIR / "knee_angular_speed.png"
plt.savefig(speed_path, dpi=200)
plt.close()

print("\nGraphs saved:")
print(angle_path)
print(speed_path)

print("\nAnalysis completed successfully.")

