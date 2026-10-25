
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# Tennis AI - Biomechanical Feature Analysis
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "runs" / "tennis_joint_angles.csv"
OUTPUT_DIR = BASE_DIR / "runs" / "risk_analysis"
OUTPUT_CSV = OUTPUT_DIR / "tennis_motion_features.csv"

FPS = 24  # غيّرها لمعدل الإطارات الحقيقي للفيديو إذا كان مختلفًا

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
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"ملف البيانات غير موجود:\n{INPUT_CSV}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required_columns = ["frame", "time_seconds"] + ANGLE_COLUMNS
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"أعمدة ناقصة في ملف CSV: {missing_columns}"
        )

    # تحويل الأعمدة الرقمية والتعامل مع القيم غير الصالحة
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["time_seconds"] = pd.to_numeric(
        df["time_seconds"], errors="coerce"
    )

    for col in ANGLE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # زوايا المفاصل المحسوبة ثنائيًا يجب أن تكون بين 0 و180 درجة
        df.loc[~df[col].between(0, 180), col] = np.nan

    df = df.sort_values("frame").reset_index(drop=True)

    # ==========================================
    # 1. حساب سرعة تغير الزوايا (درجة/ثانية)
    # ==========================================
    time_diff = df["time_seconds"].diff()

    # نتجنب القسمة على صفر أو على فرق زمني سالب
    time_diff = time_diff.where(time_diff > 0)

    for col in ANGLE_COLUMNS:
        angle_diff = df[col].diff()

        df[f"{col}_angular_speed_deg_s"] = (
            angle_diff / time_diff
        )

    # ==========================================
    # 2. الفرق بين الركبتين
    # ==========================================
    df["knee_angle_difference_deg"] = (
        df["left_knee_deg"] - df["right_knee_deg"]
    ).abs()

    # ==========================================
    # 3. مؤشر جودة القياسات
    # ==========================================
    df["valid_joint_measurements"] = (
        df[ANGLE_COLUMNS].notna().sum(axis=1)
    )

    df["missing_joint_measurements"] = (
        len(ANGLE_COLUMNS) - df["valid_joint_measurements"]
    )

    # ==========================================
    # 4. تنبيه استكشافي للقيم التي تستحق المراجعة
    # ==========================================
    # هذه حدود فحص برمجية فقط وليست حدودًا طبية.
    # يجب مراجعة الفيديو قبل تفسير أي قيمة.
    df["review_flag"] = False

    # اختلاف كبير بين زوايا الركبتين: يستحق مراجعة الإطار
    df.loc[
        df["knee_angle_difference_deg"] > 35,
        "review_flag"
    ] = True

    # تغيّر زاوي كبير بين إطارين متتاليين خلال ثانية
    for col in ANGLE_COLUMNS:
        speed_col = f"{col}_angular_speed_deg_s"
        df.loc[
            df[speed_col].abs() > 300,
            "review_flag"
        ] = True

    # ==========================================
    # 5. حفظ النتائج
    # ==========================================
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 55)
    print("TENNIS AI - MOTION FEATURE ANALYSIS")
    print("=" * 55)

    print(f"عدد الإطارات: {len(df)}")
    print(f"عدد القياسات: {len(ANGLE_COLUMNS)}")

    print("\nمتوسط الزوايا:")
    for col in ANGLE_COLUMNS:
        print(f"{col}: {df[col].mean():.2f} degrees")

    print("\nجودة القياسات:")
    print(
        "متوسط القياسات المتاحة لكل إطار:",
        f"{df['valid_joint_measurements'].mean():.2f}"
    )

    print("\nملخص المراجعة:")
    print("عدد الإطارات التي تستحق المراجعة:", int(df["review_flag"].sum()))

    print("\nتم حفظ النتائج هنا:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()

