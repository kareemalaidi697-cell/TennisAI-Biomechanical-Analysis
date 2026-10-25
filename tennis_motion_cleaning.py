
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "runs"
    / "tennis_joint_angles.csv"
)

OUTPUT_DIR = BASE_DIR / "runs" / "cleaned_motion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "cleaned_joint_angles.csv"

ANGLE_COLUMNS = [
    "left_knee_deg",
    "right_knee_deg",
    "left_hip_deg",
    "right_hip_deg",
    "left_elbow_deg",
    "right_elbow_deg",
    "left_shoulder_deg",
    "right_shoulder_deg",
]

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = ["frame", "time_seconds"] + ANGLE_COLUMNS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("frame").reset_index(drop=True)

    df["time_seconds"] = pd.to_numeric(
        df["time_seconds"], errors="coerce"
    )

    # نحتفظ بالقياسات الأصلية للمقارنة
    for col in ANGLE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        df.loc[
            ~df[col].between(0, 180),
            col
        ] = np.nan

        df[f"{col}_original"] = df[col]

        # نحدد القفزات الشاذة باستخدام نافذة محلية
        local_median = df[col].rolling(
            window=5,
            center=True,
            min_periods=3
        ).median()

        deviation = (df[col] - local_median).abs()

        # استبدال القيم التي تختلف بشدة عن الوسيط المحلي
        outlier = deviation > 30

        df.loc[outlier, col] = np.nan

        # نستخدم وسيطًا متحركًا لتقليل الاهتزاز
        df[f"{col}_clean"] = df[col].rolling(
            window=5,
            center=True,
            min_periods=1
        ).median()

    # حساب السرعات باستخدام التوقيت الفعلي
    time_diff = df["time_seconds"].diff()
    time_diff = time_diff.where(time_diff > 0)

    for col in ANGLE_COLUMNS:
        clean_col = f"{col}_clean"
        speed_col = f"{col}_angular_speed_deg_s"

        df[speed_col] = df[clean_col].diff() / time_diff

    # حفظ البيانات
    df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 55)
    print("TENNIS AI - MOTION DATA CLEANING")
    print("=" * 55)

    print(f"Frames processed: {len(df)}")

    for col in ANGLE_COLUMNS:
        original_count = df[f"{col}_original"].notna().sum()
        cleaned_count = df[f"{col}_clean"].notna().sum()

        print(f"\n{col}")
        print(f"Original valid measurements: {original_count}")
        print(f"Cleaned measurements: {cleaned_count}")

        speed_col = f"{col}_angular_speed_deg_s"
        print(
            "Maximum absolute cleaned angular speed:",
            round(df[speed_col].abs().max(), 2),
            "deg/s"
        )

    # رسم مقارنة زوايا الركبتين
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for original, clean, label in [
        (
            "left_knee_deg_original",
            "left_knee_deg_clean",
            "Left Knee"
        ),
        (
            "right_knee_deg_original",
            "right_knee_deg_clean",
            "Right Knee"
        ),
    ]:
        axes[0].plot(
            df["time_seconds"],
            df[original],
            alpha=0.4,
            label=f"{label} Original"
        )

        axes[0].plot(
            df["time_seconds"],
            df[clean],
            linewidth=1.5,
            label=f"{label} Cleaned"
        )

    axes[0].set_title("Original vs Cleaned Knee Angles")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Angle (degrees)")
    axes[0].set_ylim(0, 185)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        df["time_seconds"],
        df["left_knee_deg_angular_speed_deg_s"],
        label="Left Knee"
    )

    axes[1].plot(
        df["time_seconds"],
        df["right_knee_deg_angular_speed_deg_s"],
        label="Right Knee"
    )

    axes[1].set_title("Angular Speed After Cleaning")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Degrees/second")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    graph_path = OUTPUT_DIR / "cleaning_comparison.png"
    plt.savefig(graph_path, dpi=200)
    plt.close()

    print("\nSaved cleaned dataset:")
    print(OUTPUT_CSV)

    print("\nSaved comparison graph:")
    print(graph_path)


if __name__ == "__main__":
    main()

