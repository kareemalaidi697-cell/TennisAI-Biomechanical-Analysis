
import cv2
import csv
import math
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# TennisAI - Biomechanical Motion Analysis
# ==========================================

MODEL_PATH = "yolo11n-pose.pt"

OUTPUT_DIR = Path("runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "tennis_biomechanics.mp4"
OUTPUT_CSV = OUTPUT_DIR / "tennis_joint_angles.csv"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# COCO keypoint indices used by YOLO Pose:
# 5: Left shoulder
# 6: Right shoulder
# 7: Left elbow
# 8: Right elbow
# 9: Left wrist
# 10: Right wrist
# 11: Left hip
# 12: Right hip
# 13: Left knee
# 14: Right knee
# 15: Left ankle
# 16: Right ankle

def calculate_angle(a, b, c):
    """
    Calculate the 2D angle ABC in degrees.
    B is the joint whose angle is measured.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def get_joint_angle(keypoints, a, b, c, min_conf=0.35):
    """
    Return an angle only if all three required keypoints
    have adequate confidence.
    """
    indices = [a, b, c]

    for idx in indices:
        if float(keypoints.conf[0][idx]) < min_conf:
            return None

    points = keypoints.xy[0].cpu().numpy()

    return calculate_angle(
        points[a],
        points[b],
        points[c]
    )


def format_angle(value):
    if value is None:
        return "N/A"
    return f"{value:.1f} deg"


def main():
    print("======================================")
    print(" TennisAI Biomechanical Analyzer")
    print("======================================")
    print("Device:", "CUDA GPU" if DEVICE != "cpu" else "CPU")

    source_text = input(
        "Enter video path, or 0 for webcam: "
    ).strip().strip('"')

    source = 0 if source_text == "0" else source_text

    if isinstance(source, str) and not Path(source).is_file():
        raise FileNotFoundError(
            f"Video not found: {Path(source).resolve()}"
        )

    print("Loading YOLO Pose model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Could not open video or webcam.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 120:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "time_seconds",
        "left_knee_deg",
        "right_knee_deg",
        "left_hip_deg",
        "right_hip_deg",
        "left_elbow_deg",
        "right_elbow_deg",
        "left_shoulder_deg",
        "right_shoulder_deg"
    ])

    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model.predict(
                frame,
                device=DEVICE,
                imgsz=640,
                conf=0.35,
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            angles = {
                "left_knee": None,
                "right_knee": None,
                "left_hip": None,
                "right_hip": None,
                "left_elbow": None,
                "right_elbow": None,
                "left_shoulder": None,
                "right_shoulder": None
            }

            if result.keypoints is not None and len(result.keypoints) > 0:
                # Analyze the most confidently detected person.
                person_index = int(result.boxes.conf.argmax().item())

                # Select that person's keypoints.
                keypoints = result.keypoints[person_index:person_index + 1]

                angles["left_knee"] = get_joint_angle(
                    keypoints, 11, 13, 15
                )
                angles["right_knee"] = get_joint_angle(
                    keypoints, 12, 14, 16
                )

                angles["left_hip"] = get_joint_angle(
                    keypoints, 5, 11, 13
                )
                angles["right_hip"] = get_joint_angle(
                    keypoints, 6, 12, 14
                )

                angles["left_elbow"] = get_joint_angle(
                    keypoints, 5, 7, 9
                )
                angles["right_elbow"] = get_joint_angle(
                    keypoints, 6, 8, 10
                )

                angles["left_shoulder"] = get_joint_angle(
                    keypoints, 7, 5, 11
                )
                angles["right_shoulder"] = get_joint_angle(
                    keypoints, 8, 6, 12
                )

            # Display the measured angles on the video.
            display_items = [
                ("L Knee", angles["left_knee"]),
                ("R Knee", angles["right_knee"]),
                ("L Hip", angles["left_hip"]),
                ("R Hip", angles["right_hip"]),
                ("L Elbow", angles["left_elbow"]),
                ("R Elbow", angles["right_elbow"]),
                ("L Shoulder", angles["left_shoulder"]),
                ("R Shoulder", angles["right_shoulder"])
            ]

            y = 30

            for label, value in display_items:
                text = f"{label}: {format_angle(value)}"

                cv2.putText(
                    annotated,
                    text,
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                y += 25

            time_seconds = frame_number / fps

            csv_writer.writerow([
                frame_number,
                round(time_seconds, 4),
                angles["left_knee"],
                angles["right_knee"],
                angles["left_hip"],
                angles["right_hip"],
                angles["left_elbow"],
                angles["right_elbow"],
                angles["left_shoulder"],
                angles["right_shoulder"]
            ])

            writer.write(annotated)
            cv2.imshow("TennisAI - Biomechanical Analysis", annotated)

            frame_number += 1

            if frame_number % 100 == 0:
                print("Processed frames:", frame_number)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

    print("\nAnalysis completed!")
    print("Frames processed:", frame_number)
    print("Output video:", OUTPUT_VIDEO.resolve())
    print("Joint angles CSV:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()

